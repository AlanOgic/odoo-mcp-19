"""
Pydantic response models for the Odoo MCP Server tools.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .safety import SafetyClassification


class IssueAnalysis(BaseModel):
    """Analysis of issues encountered during execution."""
    category: str = Field(description="Error category: timeout, relational_filter, computed_field, access_rights, memory, data_integrity, unknown")
    cause: str = Field(description="Human-readable cause description")
    domain_patterns: List[str] = Field(default_factory=list, description="Detected patterns in domain that may cause issues")
    problematic_fields: List[str] = Field(default_factory=list, description="Fields that may cause issues")
    suggested_solutions: List[str] = Field(default_factory=list, description="Suggested solutions for the issue")
    model_specific_advice: List[str] = Field(default_factory=list, description="Model-specific recommendations")


class ExecuteMethodResponse(BaseModel):
    """Response model for execute_method tool with structured output."""
    success: bool = Field(description="Whether the execution was successful")
    result: Optional[Any] = Field(default=None, description="Result of the method call")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    suggestion: Optional[str] = Field(default=None, description="Helpful suggestion for fixing the error")
    hint: Optional[str] = Field(default=None, description="Additional hint for troubleshooting")
    fallback_used: bool = Field(default=False, description="Whether automatic fallback was triggered")
    issue_analysis: Optional[IssueAnalysis] = Field(default=None, description="Issue analysis when fallback was used")
    note: Optional[str] = Field(default=None, description="Additional note about the execution")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")
    pending_confirmation: bool = Field(default=False, description="Whether the operation requires confirmation before execution")
    safety: Optional[SafetyClassification] = Field(default=None, description="Safety classification of the operation")


class BatchOperationResult(BaseModel):
    """Result of a single batch operation."""
    operation_index: int = Field(description="Index of the operation in the batch")
    success: bool = Field(description="Whether this operation succeeded")
    result: Optional[Any] = Field(default=None, description="Result if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchExecuteResponse(BaseModel):
    """Response model for batch_execute tool with structured output."""
    success: bool = Field(description="Whether all operations succeeded")
    results: List[BatchOperationResult] = Field(description="Results for each operation")
    total_operations: int = Field(description="Total operations attempted")
    successful_operations: int = Field(description="Successful operations count")
    failed_operations: int = Field(description="Failed operations count")
    error: Optional[str] = Field(default=None, description="Overall error message if any operation failed")
    execution_time_ms: Optional[float] = Field(default=None, description="Total execution time in milliseconds")
    pending_confirmation: bool = Field(default=False, description="Whether the batch requires confirmation before execution")
    safety_preview: Optional[List[SafetyClassification]] = Field(default=None, description="Safety classifications for each operation")
    overall_risk: Optional[str] = Field(default=None, description="Overall risk level across all operations")


class WorkflowStepResult(BaseModel):
    """Result of a single workflow step."""
    step: str = Field(description="Name of the workflow step")
    success: bool = Field(description="Whether this step succeeded")
    skipped: bool = Field(default=False, description="Whether this step was skipped")
    reason: Optional[str] = Field(default=None, description="Reason for skipping or failure")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    result: Optional[Any] = Field(default=None, description="Step result data")


class ExecuteWorkflowResponse(BaseModel):
    """Response model for execute_workflow tool with structured output."""
    workflow: str = Field(description="Name of the executed workflow")
    success: bool = Field(description="Whether the workflow completed successfully")
    steps: List[WorkflowStepResult] = Field(default_factory=list, description="Results for each workflow step")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    available_workflows: Optional[List[str]] = Field(default=None, description="Available workflows if unknown workflow requested")
    tip: Optional[str] = Field(default=None, description="Helpful tip for using workflows")
    # Additional result fields for specific workflows
    invoice_id: Optional[int] = Field(default=None, description="Created invoice ID (for invoice workflows)")
    invoice_ids: Optional[List[int]] = Field(default=None, description="Created invoice IDs (for order workflows)")
    execution_time_ms: Optional[float] = Field(default=None, description="Total execution time in milliseconds")
    pending_confirmation: bool = Field(default=False, description="Whether the workflow requires confirmation before execution")
    safety_preview: Optional[List[SafetyClassification]] = Field(default=None, description="Safety classifications for each workflow step")
    overall_risk: Optional[str] = Field(default=None, description="Overall risk level across all workflow steps")
