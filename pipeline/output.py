"""
Phase 5: Output Formatter
Formats note events into various output formats.
"""

import json
import csv
import sys
from typing import List, Dict, Any, Optional
from io import StringIO
from pathlib import Path


class OutputFormatter:
    """Formats note events into table, CSV, JSON, or leadsheet output."""

    def __init__(self, note_events: List[Dict[str, Any]]):
        """
        Initialize formatter with note events.

        Args:
            note_events: List of note event dicts
        """
        self.note_events = note_events

    def format_table(self) -> str:
        """Format note events as a terminal table."""
        try:
            from tabulate import tabulate
        except ImportError:
            return "Error: tabulate module not installed. Please install with: pip install tabulate"

        if not self.note_events:
            return "No note events to display."

        table_data = []
        for event in self.note_events:
            row = [
                event.get('note', ''),
                f"{event.get('start_time', 0):.3f}",
                f"{event.get('end_time', 0):.3f}",
                f"{event.get('confidence', 0):.3f}",
                event.get('lyric', '') or ''
            ]
            table_data.append(row)

        headers = ["Note", "Start (s)", "End (s)", "Confidence", "Lyric"]
        return tabulate(table_data, headers=headers, tablefmt="grid")

    def format_csv(self) -> str:
        """Format note events as CSV."""
        if not self.note_events:
            return ""

        fieldnames = list(self.note_events[0].keys())
        preferred_order = ['note', 'start_time', 'end_time', 'confidence', 'lyric']
        fieldnames = [f for f in preferred_order if f in fieldnames] + \
                     [f for f in fieldnames if f not in preferred_order]

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for event in self.note_events:
            cleaned = {k: (v if v is not None else '') for k, v in event.items()}
            writer.writerow(cleaned)

        return output.getvalue()

    def format_json(self, indent: Optional[int] = 2) -> str:
        """Format note events as JSON."""
        return json.dumps(self.note_events, indent=indent, ensure_ascii=False)

    def format_leadsheet(self) -> str:
        """
        Format note events as a leadsheet mimicking a hand transcription.
        Notes are spaced with dashes to reflect relative timing gaps between them.
        """
        if not self.note_events:
            return "No note events."

        lines = []
        has_lyrics = any(event.get('lyric') for event in self.note_events)

        def space_notes(notes_with_times):
            """Insert dashes between notes based on relative time gaps."""
            if not notes_with_times:
                return ""
            if len(notes_with_times) == 1:
                return notes_with_times[0][0]

            result = []
            for i, (note, time) in enumerate(notes_with_times):
                result.append(note)
                if i < len(notes_with_times) - 1:
                    gap = notes_with_times[i + 1][1] - time
                    # Scale: < 0.3s = 1 space, 0.3-0.8s = " - ", 0.8-2s = " -- ", > 2s = " --- "
                    if gap < 0.3:
                        result.append(" ")
                    elif gap < 0.8:
                        result.append(" - ")
                    elif gap < 2.0:
                        result.append(" -- ")
                    else:
                        result.append(" --- ")
            return "".join(result)

        if has_lyrics:
            current_lyric = None
            current_notes = []

            for event in self.note_events:
                lyric = event.get('lyric')
                if lyric != current_lyric:
                    if current_notes:
                        if current_lyric:
                            lines.append(current_lyric)
                        lines.append(space_notes(current_notes))
                        lines.append("")
                    current_lyric = lyric
                    current_notes = [(event['note'], event['start_time'])]
                else:
                    current_notes.append((event['note'], event['start_time']))

            if current_notes:
                if current_lyric:
                    lines.append(current_lyric)
                lines.append(space_notes(current_notes))

        else:
            current_group = []
            prev_time = None

            for event in self.note_events:
                if prev_time is not None and (event['start_time'] - prev_time) > 2.0:
                    if current_group:
                        lines.append(space_notes(current_group))
                        current_group = []
                current_group.append((event['note'], event['start_time']))
                prev_time = event['start_time']

            if current_group:
                lines.append(space_notes(current_group))

        return "\n".join(lines)

    def output(self, format_type: str = "table",
               output_file: Optional[str] = None) -> Optional[str]:
        """
        Output note events in specified format.

        Args:
            format_type: "table", "csv", "json", or "leadsheet"
            output_file: Optional file path to write output to

        Returns:
            Formatted string if no output_file, None otherwise

        Raises:
            ValueError: If format_type is invalid
        """
        format_type = format_type.lower()

        if format_type == "table":
            formatted = self.format_table()
        elif format_type == "csv":
            formatted = self.format_csv()
        elif format_type == "json":
            formatted = self.format_json()
        elif format_type == "leadsheet":
            formatted = self.format_leadsheet()
        else:
            raise ValueError(f"Invalid format type: {format_type}. Must be 'table', 'csv', 'json', or 'leadsheet'.")

        if output_file:
            output_path = Path(output_file).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted)

            print(f"Output written to: {output_path}")
            return None
        else:
            return formatted

    def print_table(self) -> None:
        print(self.format_table())

    def print_csv(self) -> None:
        print(self.format_csv())

    def print_json(self) -> None:
        print(self.format_json())

    def print_leadsheet(self) -> None:
        print(self.format_leadsheet())


def format_output(note_events: List[Dict[str, Any]],
                  format_type: str = "table",
                  output_file: Optional[str] = None) -> Optional[str]:
    """
    Convenience function for formatting output.

    Args:
        note_events: List of note event dicts
        format_type: "table", "csv", "json", or "leadsheet"
        output_file: Optional file path to write output to

    Returns:
        Formatted string if no output_file, None otherwise
    """
    formatter = OutputFormatter(note_events)
    return formatter.output(format_type, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format note events for output")
    parser.add_argument("input", help="Input JSON file with note events")
    parser.add_argument("--format", "-f",
                        choices=["table", "csv", "json", "leadsheet"],
                        default="table", help="Output format")
    parser.add_argument("--output", "-o", help="Output file (optional)")

    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            note_events = json.load(f)

        result = format_output(note_events, args.format, args.output)

        if result is not None:
            print(result)

    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)