"""
Test package initialization.
"""

def test_imports():
    """Test that all main imports work correctly."""
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        
        from src.data import create_dataloaders
        from src.models import RepresentationLayer, ConvDecoder, GaussianMixture
        from src.training import DGDTrainer
        from src.optimization import OptunaOptimizer
        from src.visualization import LatentSpaceVisualizer
        
        print("All imports successful!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False


if __name__ == "__main__":
    test_imports()
