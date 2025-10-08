# File: setup_project.py
"""
Script to create the project structure for centrifugal compressor design
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete directory structure"""
    
    # Define project structure
    structure = {
        'centrifugal_compressor': {
            'core': [
                '__init__.py',
                'geometry.py',
                'thermodynamics.py',
                'velocity.py',
                'correlations.py'
            ],
            'losses': [
                '__init__.py',
                'loss_models.py',
                'loss_calculator.py'
            ],
            'components': [
                '__init__.py',
                'inducer.py',
                'impeller.py',
                'diffuser.py',
                'stage.py'
            ],
            'analysis': [
                '__init__.py',
                'performance.py',
                'off_design.py',
                'constraints.py'
            ],
            'optimization': [
                '__init__.py',
                'design_variables.py',
                'objectives.py',
                'optimizer.py',
                'multi_point.py'
            ],
            'visualization': [
                '__init__.py',
                'meridional.py',
                'impeller_3d.py',
                'performance_plots.py',
                'export.py'
            ]
        },
        'tests': {
            'unit': [
                '__init__.py',
                'test_thermodynamics.py',
                'test_velocity.py',
                'test_correlations.py',
                'test_losses.py'
            ],
            'integration': [
                '__init__.py',
                'test_components.py',
                'test_stage.py',
                'test_optimization.py'
            ],
            'validation': [
                '__init__.py',
                'test_radcomp_comparison.py',
                'test_turboflow_comparison.py'
            ],
            'fixtures': [
                '__init__.py',
                'test_data.py',
                'reference_values.py'
            ]
        },
        'examples': [
            'example_analysis.py',
            'example_optimization.py',
            'example_visualization.py'
        ],
        'docs': [
            'README.md',
            'API.md',
            'THEORY.md'
        ]
    }
    
    def create_structure(base_path, struct):
        """Recursively create directory structure"""
        for name, content in struct.items():
            path = base_path / name
            path.mkdir(exist_ok=True)
            
            if isinstance(content, dict):
                create_structure(path, content)
            elif isinstance(content, list):
                for file in content:
                    (path / file).touch()
    
    # Create structure
    base = Path('.')
    create_structure(base, structure)
    
    # Create root files
    (base / 'setup.py').touch()
    (base / 'requirements.txt').touch()
    (base / 'README.md').touch()
    (base / '.gitignore').touch()
    
    print("✓ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start building from Layer 1: core/thermodynamics.py")

if __name__ == "__main__":
    create_project_structure()
