"""
Conversation State Module (Phase 10A).

Provides bounded, safe, and immutable data structures for maintaining
multi-turn analytical conversation memory across questions, generated SQL,
query results, and AI insights.

ARCHITECTURAL PRINCIPLES:
- Pure State/Data Structure Module: Contains zero LLM/Gemini logic, zero SQL execution,
  zero DuckDB connections, and zero Streamlit dependencies.
- Compact & Serializable Storage: Query results are snapshotted into a lightweight
  summary representation (QueryResultSummary) instead of retaining live DataFrame references.
- Strict Memory Boundedness: Conversation history is capped (default: 10 turns),
  and result rows stored in memory are capped (default: 25 rows) to avoid memory leaks
  or exploding token context.
- Immutability Protection: DataFrames passed in are deep-copied into independent
  records, ensuring external modifications never corrupt conversational state.
- Dataset Boundaries: History is tied to the active dataset and can be explicitly
  cleared when a new dataset is loaded.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import copy
import pandas as pd


@dataclass(frozen=True)
class QueryResultSummary:
    """
    Compact, immutable, and serializable snapshot of an analytical query result.

    Attributes:
        columns: List of column names in the result.
        rows: Truncated list of row records (list of dicts).
        row_count: Original total number of rows produced by the query.
        is_truncated: True if original row count exceeds max_result_rows.
        truncated_rows: Count of rows actually stored in `rows`.
    """
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    is_truncated: bool
    truncated_rows: int

    @property
    def empty(self) -> bool:
        """Returns True if the result produced zero rows."""
        return self.row_count == 0

    @classmethod
    def from_dataframe(
        cls,
        df: Optional[pd.DataFrame],
        max_rows: int = 25,
    ) -> "QueryResultSummary":
        """
        Creates a QueryResultSummary from a Pandas DataFrame with row truncation
        and deep-copy immutability protection.
        """
        if df is None:
            return cls(
                columns=[],
                rows=[],
                row_count=0,
                is_truncated=False,
                truncated_rows=0,
            )

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame or None, got {type(df).__name__}")

        total_rows = len(df)
        cols = [str(c) for c in df.columns]

        if df.empty or total_rows == 0:
            return cls(
                columns=cols,
                rows=[],
                row_count=0,
                is_truncated=False,
                truncated_rows=0,
            )

        is_truncated = total_rows > max_rows
        # Deep-copy slice to protect against external mutations
        view_df = df.iloc[:max_rows].copy(deep=True)
        # Convert to serializable record dicts and deepcopy
        raw_records = view_df.to_dict(orient="records")
        copied_records = copy.deepcopy(raw_records)

        return cls(
            columns=cols,
            rows=copied_records,
            row_count=total_rows,
            is_truncated=is_truncated,
            truncated_rows=len(copied_records),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns a serializable dictionary representation."""
        return {
            "columns": list(self.columns),
            "rows": copy.deepcopy(self.rows),
            "row_count": self.row_count,
            "is_truncated": self.is_truncated,
            "truncated_rows": self.truncated_rows,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResultSummary":
        """Reconstructs a QueryResultSummary from a dictionary representation."""
        return cls(
            columns=list(data.get("columns", [])),
            rows=copy.deepcopy(data.get("rows", [])),
            row_count=int(data.get("row_count", 0)),
            is_truncated=bool(data.get("is_truncated", False)),
            truncated_rows=int(data.get("truncated_rows", 0)),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Reconstructs a Pandas DataFrame from the stored rows.
        Note: Contains only up to max_result_rows rows.
        """
        if not self.rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.rows, columns=self.columns)


@dataclass
class ConversationTurn:
    """
    Represents a single analytical exchange between the user and the assistant.

    Attributes:
        question: The natural-language business question asked by the user.
        sql: The read-only SQL query generated for the question.
        result: The compact QueryResultSummary of the query execution.
        insights: AI-generated business insights derived from the query result.
    """
    question: str
    sql: Optional[str] = None
    result: Optional[QueryResultSummary] = None
    insights: Optional[str] = None
    sql_explanation: Optional[str] = None

    @property
    def result_summary(self) -> Optional[QueryResultSummary]:
        """Alias for result."""
        return self.result

    @property
    def has_sql(self) -> bool:
        return bool(self.sql and self.sql.strip())

    @property
    def has_result(self) -> bool:
        return self.result is not None and not self.result.empty

    @property
    def has_insights(self) -> bool:
        return bool(self.insights and self.insights.strip())

    @property
    def has_sql_explanation(self) -> bool:
        return bool(self.sql_explanation and self.sql_explanation.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the turn into a standard dictionary."""
        return {
            "question": self.question,
            "sql": self.sql,
            "result": self.result.to_dict() if self.result is not None else None,
            "insights": self.insights,
            "sql_explanation": self.sql_explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Deserializes a turn from a dictionary."""
        result_data = data.get("result")
        result_summary = QueryResultSummary.from_dict(result_data) if result_data is not None else None
        return cls(
            question=str(data.get("question", "")),
            sql=data.get("sql"),
            result=result_summary,
            insights=data.get("insights"),
            sql_explanation=data.get("sql_explanation"),
        )


class ConversationHistory:
    """
    Bounded, chronologically ordered manager for analytical conversation turns.

    Tied to a specific dataset boundary. Ensures that memory usage stays predictable
    and that caller mutations do not leak into conversation history.
    """
    def __init__(
        self,
        max_turns: int = 10,
        max_result_rows: int = 25,
        dataset_name: Optional[str] = None,
    ):
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns}")
        if max_result_rows < 0:
            raise ValueError(f"max_result_rows cannot be negative, got {max_result_rows}")

        self.max_turns = max_turns
        self.max_result_rows = max_result_rows
        self.dataset_name = dataset_name
        self._turns: List[ConversationTurn] = []

    def add_turn(
        self,
        question: Union[str, ConversationTurn],
        sql: Optional[str] = None,
        result_df: Optional[pd.DataFrame] = None,
        insights: Optional[str] = None,
        result: Optional[Union[pd.DataFrame, QueryResultSummary]] = None,
        sql_explanation: Optional[str] = None,
    ) -> ConversationTurn:
        """
        Appends a new turn to the conversation history.

        Enforces:
        - Result summarization and row truncation (max_result_rows).
        - Deep-copy immutability.
        - History boundedness (max_turns), keeping the most recent turns in chronological order.
        """
        if isinstance(question, ConversationTurn):
            # Cloning existing turn with truncation safeguards
            turn_question = question.question
            turn_sql = question.sql
            turn_insights = question.insights
            turn_sql_explanation = question.sql_explanation
            if question.result is not None:
                turn_result = copy.deepcopy(question.result)
            else:
                turn_result = None
        else:
            if not isinstance(question, str):
                raise TypeError(f"question must be a str or ConversationTurn, got {type(question).__name__}")

            turn_question = question
            turn_sql = sql
            turn_insights = insights
            turn_sql_explanation = sql_explanation

            # Determine query result source (result_df or result)
            raw_res = result_df if result_df is not None else result
            if isinstance(raw_res, QueryResultSummary):
                turn_result = copy.deepcopy(raw_res)
            elif isinstance(raw_res, pd.DataFrame):
                turn_result = QueryResultSummary.from_dataframe(raw_res, max_rows=self.max_result_rows)
            elif raw_res is None:
                turn_result = None
            else:
                raise TypeError(
                    f"result must be a pandas DataFrame, QueryResultSummary, or None, got {type(raw_res).__name__}"
                )

        new_turn = ConversationTurn(
            question=turn_question,
            sql=turn_sql,
            result=turn_result,
            insights=turn_insights,
            sql_explanation=turn_sql_explanation,
        )

        self._turns.append(new_turn)

        # Enforce bounded history size
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

        return new_turn

    def get_turns(self) -> List[ConversationTurn]:
        """
        Returns all turns in chronological order.
        Returns a shallow copy of the list to prevent external mutation of the internal list.
        """
        return list(self._turns)

    def get_recent_turns(self, limit: Optional[int] = None) -> List[ConversationTurn]:
        """
        Retrieves the most recent N turns in chronological order.

        If limit is None or >= total turns, returns all turns.
        If limit <= 0, returns an empty list.
        """
        if limit is None or limit >= len(self._turns):
            return list(self._turns)
        if limit <= 0:
            return []
        return list(self._turns[-limit:])

    def clear(self) -> None:
        """Clears all conversation turns from memory."""
        self._turns.clear()

    def reset_for_dataset(self, dataset_name: Optional[str] = None) -> None:
        """
        Resets conversation history for a new dataset boundary.
        Clears all stored turns and updates dataset_name.
        """
        self.clear()
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        """Returns the number of turns currently stored."""
        return len(self._turns)

    def is_empty(self) -> bool:
        """Returns True if the history contains no turns."""
        return len(self._turns) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the conversation history to a dictionary."""
        return {
            "max_turns": self.max_turns,
            "max_result_rows": self.max_result_rows,
            "dataset_name": self.dataset_name,
            "turns": [turn.to_dict() for turn in self._turns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """Reconstructs a ConversationHistory object from serialized data."""
        history = cls(
            max_turns=int(data.get("max_turns", 10)),
            max_result_rows=int(data.get("max_result_rows", 25)),
            dataset_name=data.get("dataset_name"),
        )
        for turn_data in data.get("turns", []):
            history.add_turn(ConversationTurn.from_dict(turn_data))
        return history
