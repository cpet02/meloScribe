"""
Phase 1: Audio Stem Separation using Demucs
Isolates vocals from an MP3 file using the Demucs model.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import warnings
import sys

class Stemmer:
    """Wrapper for Demucs vocal isolation."""
    
    def __init__(self, model_name="htdemucs", device=None):
        """
        Initialize the stemmer.
        
        Args:
            model_name (str): Demucs model to use. Default: "htdemucs"
            device (str): Device to run on ("cpu", "cuda", "mps"). Auto-detects if None.
        """
        self.model_name = model_name
        
        if device is None:
            # Auto-detect device
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
    def separate_vocals(self, input_path, output_dir=None, force_reprocess=False):
        """
        Separate vocals from an MP3 file using Demucs.
        
        Args:
            input_path (str): Path to input MP3 file
            output_dir (str, optional): Directory to save outputs. If None, creates temp dir.
            force_reprocess (bool): If True, reprocess even if output exists
            
        Returns:
            str: Path to isolated vocals.wav file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input file is not an MP3
            RuntimeError: If Demucs fails to process the file
        """
        # Validate input
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() != '.mp3':
            raise ValueError(f"Input file must be MP3. Got: {input_path.suffix}")
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="demucs_", dir=input_path.parent))
        else:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model-specific output path
        model_output_dir = output_dir / self.model_name
        
        # Check if vocals already exist
        vocals_path = model_output_dir / Path(input_path.stem) / "vocals.wav"
        
        if vocals_path.exists() and not force_reprocess:
            print(f"Using cached vocals: {vocals_path}")
            return str(vocals_path)
        
        print(f"Separating vocals from: {input_path.name}")
        print(f"Using model: {self.model_name}, device: {self.device}")
        
        try:
            # Run Demucs separation
            cmd = [
                "demucs",
                "--name", self.model_name,
                "--device", self.device,
                "--out", str(output_dir),
                str(input_path)
            ]
            
            # Suppress warnings from subprocess
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            if result.returncode != 0:
                raise RuntimeError(f"Demucs failed: {result.stderr}")
            
            # Verify output file exists
            if not vocals_path.exists():
                # Try alternative path structure (some Demucs versions differ)
                possible_paths = [
                    vocals_path,
                    model_output_dir / "htdemucs" / Path(input_path.stem) / "vocals.wav",
                    output_dir / "htdemucs" / Path(input_path.stem) / "vocals.wav",
                ]
                
                for path in possible_paths:
                    if path.exists():
                        vocals_path = path
                        break
                else:
                    raise FileNotFoundError(f"Vocals file not found in expected location: {vocals_path}")
            
            print(f"Vocals isolated: {vocals_path}")
            return str(vocals_path)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Demucs process failed with code {e.returncode}: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Demucs command not found. Please ensure demucs is installed: pip install demucs")
    
    def cleanup_temp_files(self, output_dir):
        """
        Clean up temporary output directory.
        
        Args:
            output_dir (str): Directory to clean up
        """
        output_dir = Path(output_dir)
        if output_dir.exists() and "demucs_" in output_dir.name:  # Only clean our temp dirs
            try:
                shutil.rmtree(output_dir)
                print(f"Cleaned up temporary directory: {output_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {output_dir}: {e}")


def isolate_vocals(input_path, output_dir=None, force_reprocess=False):
    """
    Convenience function for isolating vocals.
    
    Args:
        input_path (str): Path to input MP3 file
        output_dir (str, optional): Directory to save outputs
        force_reprocess (bool): If True, reprocess even if output exists
        
    Returns:
        str: Path to isolated vocals.wav file
    """
    stemmer = Stemmer()
    return stemmer.separate_vocals(input_path, output_dir, force_reprocess)


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Isolate vocals from MP3 using Demucs")
    parser.add_argument("input", help="Path to input MP3 file")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: auto)")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing")
    
    args = parser.parse_args()
    
    try:
        vocals_path = isolate_vocals(args.input, args.output_dir, args.force)
        print(f"\nSuccess! Vocals saved to: {vocals_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)