import os
import shutil
from pathlib import Path
from typing import List, Set
import logging
from ..utils.logging_config import logger

class ProjectCleanup:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.used_files: Set[Path] = set()
        self.unused_files: List[Path] = []

    def scan_imports(self, file_path: Path) -> None:
        """Scan Python files for imports and track used files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Add the current file to used files
                self.used_files.add(file_path)
                
                # Scan for imports
                lines = content.split('\n')
                for line in lines:
                    if line.startswith(('import ', 'from ')):
                        # Basic import parsing (can be enhanced)
                        import_path = line.split()[1].split('.')[0]
                        potential_path = self.project_root / f"{import_path}.py"
                        if potential_path.exists():
                            self.used_files.add(potential_path)
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {str(e)}")

    def find_unused_files(self) -> None:
        """Find unused Python files in the project."""
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if file_path not in self.used_files:
                        self.unused_files.append(file_path)

    def remove_unused_files(self, dry_run: bool = True) -> None:
        """Remove unused files (with dry run option)."""
        for file_path in self.unused_files:
            if dry_run:
                logger.info(f"Would remove: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")

    def cleanup_empty_directories(self, dry_run: bool = True) -> None:
        """Remove empty directories (with dry run option)."""
        for root, dirs, _ in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if not any(dir_path.iterdir()):
                    if dry_run:
                        logger.info(f"Would remove empty directory: {dir_path}")
                    else:
                        try:
                            dir_path.rmdir()
                            logger.info(f"Removed empty directory: {dir_path}")
                        except Exception as e:
                            logger.error(f"Error removing directory {dir_path}: {str(e)}")

def main():
    # Initialize cleanup
    cleanup = ProjectCleanup("UnMask")
    
    # Scan all Python files
    for root, _, files in os.walk("UnMask"):
        for file in files:
            if file.endswith('.py'):
                cleanup.scan_imports(Path(root) / file)
    
    # Find and report unused files
    cleanup.find_unused_files()
    logger.info(f"Found {len(cleanup.unused_files)} unused files")
    
    # Run cleanup (dry run first)
    cleanup.remove_unused_files(dry_run=True)
    cleanup.cleanup_empty_directories(dry_run=True)
    
    # Ask for confirmation before actual removal
    response = input("Proceed with actual file removal? (y/n): ")
    if response.lower() == 'y':
        cleanup.remove_unused_files(dry_run=False)
        cleanup.cleanup_empty_directories(dry_run=False)

if __name__ == "__main__":
    main() 