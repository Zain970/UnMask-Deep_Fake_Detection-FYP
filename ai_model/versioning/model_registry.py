import torch
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import shutil
import logging
import os
from dataclasses import dataclass, asdict
import yaml

@dataclass
class ModelVersion:
    """Class representing a model version."""
    version: str
    model_path: Path
    metadata: Dict
    created_at: datetime
    hash: str
    parent_version: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None

class ModelRegistry:
    def __init__(
        self,
        registry_path: Union[str, Path],
        model_store_path: Union[str, Path]
    ):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to store registry metadata
            model_store_path: Path to store model files
        """
        self.registry_path = Path(registry_path)
        self.model_store_path = Path(model_store_path)
        
        # Create directories if they don't exist
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.model_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry
        self.versions: Dict[str, ModelVersion] = {}
        self._load_registry()
        
        # Set up logging
        self.logger = logging.getLogger('ModelRegistry')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.registry_path / 'registry.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def _load_registry(self) -> None:
        """Load registry data from disk."""
        registry_file = self.registry_path / 'registry.json'
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for version_data in data['versions']:
                    version = ModelVersion(
                        version=version_data['version'],
                        model_path=Path(version_data['model_path']),
                        metadata=version_data['metadata'],
                        created_at=datetime.fromisoformat(version_data['created_at']),
                        hash=version_data['hash'],
                        parent_version=version_data.get('parent_version'),
                        description=version_data.get('description'),
                        tags=version_data.get('tags', [])
                    )
                    self.versions[version.version] = version
    
    def _save_registry(self) -> None:
        """Save registry data to disk."""
        registry_file = self.registry_path / 'registry.json'
        data = {
            'versions': [
                {
                    'version': v.version,
                    'model_path': str(v.model_path),
                    'metadata': v.metadata,
                    'created_at': v.created_at.isoformat(),
                    'hash': v.hash,
                    'parent_version': v.parent_version,
                    'description': v.description,
                    'tags': v.tags
                }
                for v in self.versions.values()
            ]
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_model_hash(self, model: torch.nn.Module) -> str:
        """Calculate hash of model weights."""
        state_dict = model.state_dict()
        hash_object = hashlib.sha256()
        for key, value in state_dict.items():
            hash_object.update(key.encode())
            hash_object.update(value.numpy().tobytes())
        return hash_object.hexdigest()
    
    def register_model(
        self,
        model: torch.nn.Module,
        version: str,
        metadata: Dict,
        parent_version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model: The model to register
            version: Version string (e.g., '1.0.0')
            metadata: Dictionary of model metadata
            parent_version: Optional parent version
            description: Optional description
            tags: Optional list of tags
        
        Returns:
            ModelVersion object
        """
        # Check if version already exists
        if version in self.versions:
            raise ValueError(f"Version {version} already exists")
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model)
        
        # Create version directory
        version_dir = self.model_store_path / version
        version_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = version_dir / 'model.pt'
        torch.save(model.state_dict(), model_path)
        
        # Create version object
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            metadata=metadata,
            created_at=datetime.now(),
            hash=model_hash,
            parent_version=parent_version,
            description=description,
            tags=tags or []
        )
        
        # Add to registry
        self.versions[version] = model_version
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Registered model version {version}")
        
        return model_version
    
    def get_version(self, version: str) -> ModelVersion:
        """Get a specific model version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        return self.versions[version]
    
    def load_model(self, version: str) -> torch.nn.Module:
        """Load a model from a specific version."""
        model_version = self.get_version(version)
        
        # Create model instance
        model = torch.nn.Module()  # Replace with actual model class
        model.load_state_dict(torch.load(model_version.model_path))
        
        return model
    
    def list_versions(
        self,
        tags: Optional[List[str]] = None,
        include_metadata: bool = False
    ) -> List[Dict]:
        """
        List all model versions, optionally filtered by tags.
        
        Args:
            tags: Optional list of tags to filter by
            include_metadata: Whether to include metadata in output
        
        Returns:
            List of version dictionaries
        """
        versions = self.versions.values()
        
        # Filter by tags if provided
        if tags:
            versions = [
                v for v in versions
                if any(tag in v.tags for tag in tags)
            ]
        
        # Convert to dictionaries
        result = []
        for version in versions:
            version_dict = {
                'version': version.version,
                'created_at': version.created_at,
                'description': version.description,
                'tags': version.tags,
                'parent_version': version.parent_version
            }
            if include_metadata:
                version_dict['metadata'] = version.metadata
            result.append(version_dict)
        
        return result
    
    def delete_version(self, version: str) -> None:
        """Delete a model version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Remove model files
        version_dir = self.model_store_path / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from registry
        del self.versions[version]
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Deleted model version {version}")
    
    def export_version(
        self,
        version: str,
        output_path: Union[str, Path]
    ) -> None:
        """
        Export a model version to a specified path.
        
        Args:
            version: Version to export
            output_path: Path to export to
        """
        model_version = self.get_version(version)
        output_path = Path(output_path)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        shutil.copy2(
            model_version.model_path,
            output_path / 'model.pt'
        )
        
        # Save version info
        version_info = {
            'version': model_version.version,
            'metadata': model_version.metadata,
            'created_at': model_version.created_at.isoformat(),
            'hash': model_version.hash,
            'parent_version': model_version.parent_version,
            'description': model_version.description,
            'tags': model_version.tags
        }
        
        with open(output_path / 'version_info.yaml', 'w') as f:
            yaml.dump(version_info, f)
        
        self.logger.info(f"Exported model version {version} to {output_path}")
    
    def import_version(
        self,
        import_path: Union[str, Path]
    ) -> ModelVersion:
        """
        Import a model version from a specified path.
        
        Args:
            import_path: Path containing exported version
        
        Returns:
            Imported ModelVersion object
        """
        import_path = Path(import_path)
        
        # Load version info
        with open(import_path / 'version_info.yaml', 'r') as f:
            version_info = yaml.safe_load(f)
        
        # Check if version already exists
        if version_info['version'] in self.versions:
            raise ValueError(f"Version {version_info['version']} already exists")
        
        # Create version directory
        version_dir = self.model_store_path / version_info['version']
        version_dir.mkdir(exist_ok=True)
        
        # Copy model file
        shutil.copy2(
            import_path / 'model.pt',
            version_dir / 'model.pt'
        )
        
        # Create version object
        model_version = ModelVersion(
            version=version_info['version'],
            model_path=version_dir / 'model.pt',
            metadata=version_info['metadata'],
            created_at=datetime.fromisoformat(version_info['created_at']),
            hash=version_info['hash'],
            parent_version=version_info.get('parent_version'),
            description=version_info.get('description'),
            tags=version_info.get('tags', [])
        )
        
        # Add to registry
        self.versions[model_version.version] = model_version
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Imported model version {model_version.version}")
        
        return model_version 