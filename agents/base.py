from typing import Dict, List, Any
from abc import ABC, abstractmethod
from langchain_core.messages import AnyMessage

class BaseAgent(ABC):
  """Common interface for specialist agents."""
  name: str

  @abstractmethod
  def run(self, session_id: str, messages: List[AnyMessage]) -> Dict[str, Any]:
    """
    Execute agent logic for this turn.
    Returns:
      {
        "new_messages": [Message, ...]   # only messages produced this turn
        "tool_groups": {type: [result, ...]},
        "nl_text": str                    # agent's freeform text for this turn
      }
    """
    ...
