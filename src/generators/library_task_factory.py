"""
Library Task Factory for Code Output Prediction

This factory creates library-focused code generation tasks with varying difficulty
levels, integrating the library generator, executor, and verifier components.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .library_generator import LibraryCodeGenerator, DifficultyLevel
from ..executors.library_executor import LibraryCodeExecutor


@dataclass
class TaskConfiguration:
    """Configuration for library task generation."""
    libraries: List[str]
    difficulty_levels: List[DifficultyLevel]
    min_tasks_per_library: int = 5
    max_tasks_per_library: int = 20
    timeout_seconds: int = 30
    enable_progressive_difficulty: bool = True
    require_all_libraries: bool = False


class LibraryTaskFactory:
    """
    Factory for creating comprehensive library-based code output prediction tasks.
    
    This factory orchestrates the entire pipeline:
    1. Generate library-specific code with varying difficulty
    2. Execute code safely with proper library support
    3. Verify outputs with enhanced verification methods
    4. Provide detailed feedback and scoring
    """
    
    def __init__(self, api_key: str, config_path: str = "config.yaml"):
        """
        Initialize the library task factory.
        
        Args:
            api_key: OpenAI API key for code generation
            config_path: Path to configuration file
        """
        self.generator = LibraryCodeGenerator(api_key, config_path)
        self.executor = LibraryCodeExecutor(timeout_seconds=30, max_memory_mb=512)
        
        # Check library availability
        self.library_status = self.executor.get_library_status()
        
        # Available libraries for task generation
        self.available_libraries = [
            lib for lib, available in self.library_status["available_libraries"].items()
            if available
        ]
        
        if not self.available_libraries:
            print("Warning: No libraries are available. Please install required libraries.")
    
    def create_task_suite(self, config: TaskConfiguration) -> Dict[str, Any]:
        """
        Create a comprehensive suite of library tasks.
        
        Args:
            config: Task configuration specifying libraries, difficulty, etc.
            
        Returns:
            Dict containing generated tasks, execution results, and metadata
        """
        # Filter libraries based on availability
        target_libraries = []
        for lib in config.libraries:
            if lib in self.available_libraries:
                target_libraries.append(lib)
            elif config.require_all_libraries:
                raise ValueError(f"Required library '{lib}' is not available")
            else:
                print(f"Warning: Library '{lib}' not available, skipping...")
        
        if not target_libraries:
            raise ValueError("No target libraries are available")
        
        # Generate tasks for each library and difficulty level
        all_tasks = []
        task_summary = {
            "total_tasks": 0,
            "successful_generations": 0,
            "successful_executions": 0,
            "verification_results": {},
            "library_coverage": {},
            "difficulty_distribution": {}
        }
        
        for library in target_libraries:
            library_tasks = self._create_library_tasks(library, config)
            all_tasks.extend(library_tasks)
            
            # Update summary
            task_summary["library_coverage"][library] = len(library_tasks)
        
        # Execute and verify all tasks
        for i, task in enumerate(all_tasks):
            print(f"Processing task {i+1}/{len(all_tasks)}: {task['library']} ({task['difficulty']})")
            
            # Execute the generated code
            execution_result = self._execute_task(task)
            task["execution_result"] = execution_result
            
            # Verify the output
            verification_result = self._verify_task_output(task, execution_result)
            task["verification_result"] = verification_result
            
            # Update summary statistics
            self._update_task_summary(task_summary, task, execution_result, verification_result)
        
        return {
            "tasks": all_tasks,
            "summary": task_summary,
            "library_status": self.library_status,
            "configuration": config.__dict__
        }
    
    def create_single_task(self, library: str = None, difficulty: DifficultyLevel = None) -> Dict[str, Any]:
        """
        Create a single library task with execution.
        
        Args:
            library: Specific library to use (random if None)
            difficulty: Specific difficulty level (random if None)
            
        Returns:
            Complete task result with generation and execution
        """
        # Validate library availability
        if library and library not in self.available_libraries:
            raise ValueError(f"Library '{library}' is not available")
        
        if library is None:
            if not self.available_libraries:
                raise ValueError("No libraries are available")
            library = random.choice(self.available_libraries)
        
        # Generate the code task
        generation_result = self.generator.generate_code(library, difficulty)
        
        if not generation_result["success"]:
            return {
                "success": False,
                "error": generation_result["error"],
                "stage": "generation"
            }
        
        # Execute the code
        execution_result = self._execute_task(generation_result)
        
        return {
            "success": execution_result["success"],
            "generation": generation_result,
            "execution": execution_result
        }
    
    def create_progressive_difficulty_suite(self, library: str, num_tasks_per_level: int = 3) -> Dict[str, Any]:
        """
        Create a suite of tasks with progressive difficulty for a specific library.
        
        Args:
            library: Target library
            num_tasks_per_level: Number of tasks per difficulty level
            
        Returns:
            Suite of tasks organized by difficulty level
        """
        if library not in self.available_libraries:
            raise ValueError(f"Library '{library}' is not available")
        
        suite = {
            "library": library,
            "difficulty_levels": {},
            "progression_analysis": {}
        }
        
        for difficulty in DifficultyLevel:
            level_tasks = []
            
            for i in range(num_tasks_per_level):
                task = self.create_single_task(library, difficulty)
                level_tasks.append(task)
            
            suite["difficulty_levels"][difficulty.name] = level_tasks
            
            # Analyze difficulty progression
            suite["progression_analysis"][difficulty.name] = self._analyze_difficulty_level(level_tasks)
        
        return suite
    
    def _create_library_tasks(self, library: str, config: TaskConfiguration) -> List[Dict[str, Any]]:
        """Create tasks for a specific library."""
        tasks = []
        num_tasks = random.randint(config.min_tasks_per_library, config.max_tasks_per_library)
        
        for _ in range(num_tasks):
            # Select difficulty level
            if config.enable_progressive_difficulty:
                difficulty = random.choice(config.difficulty_levels)
            else:
                difficulty = random.choice(list(DifficultyLevel))
            
            # Generate task
            task = self.generator.generate_code(library, difficulty)
            if task["success"]:
                tasks.append(task)
        
        return tasks
    
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generated task."""
        if not task.get("success", False):
            return {"success": False, "error": "Task generation failed"}
        
        code = task["code"]
        library = task["library"]
        test_inputs = task.get("test_inputs", [])
        
        # Execute with each test input
        execution_results = []
        
        if test_inputs:
            for test_input in test_inputs:
                result = self.executor.execute(
                    code, 
                    test_input["value"], 
                    required_libraries=[library]
                )
                result["test_description"] = test_input["description"]
                execution_results.append(result)
        else:
            # Execute without input
            result = self.executor.execute(code, required_libraries=[library])
            result["test_description"] = "No input test"
            execution_results.append(result)
        
        # Determine overall execution success
        any_success = any(result["success"] for result in execution_results)
        
        return {
            "success": any_success,
            "results": execution_results,
            "library_analysis": execution_results[0].get("library_usage", {}) if execution_results else {}
        }
    

    

    
    def get_available_libraries(self) -> List[str]:
        """Get list of available libraries for task generation."""
        return self.available_libraries.copy()
    
    def get_library_status(self) -> Dict[str, Any]:
        """Get detailed status of library availability."""
        return self.library_status
    
    def install_missing_libraries(self, libraries: List[str]) -> Dict[str, Any]:
        """Attempt to install missing libraries."""
        return self.executor.install_missing_libraries(libraries)
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate the complete setup and provide recommendations."""
        validation = {
            "generator_ready": True,
            "executor_ready": len(self.available_libraries) > 0,
            "verifier_ready": True,
            "available_libraries": self.available_libraries,
            "missing_libraries": self.library_status["missing_libraries"],
            "recommendations": []
        }
        
        if not validation["executor_ready"]:
            validation["recommendations"].append(
                "Install required libraries using: pip install pandas numpy requests"
            )
        
        if len(self.available_libraries) < 3:
            validation["recommendations"].append(
                f"Consider installing more libraries for diverse tasks. Currently available: {len(self.available_libraries)}"
            )
        
        validation["overall_ready"] = all([
            validation["generator_ready"],
            validation["executor_ready"],
            validation["verifier_ready"]
        ])
        
        return validation
