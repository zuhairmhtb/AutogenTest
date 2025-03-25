from typing import Dict, List, Callable, Optional, Union, Tuple
from dataclasses import dataclass
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler

@dataclass
class TextSegment:
    """
    Represents a segment of text with its start and end indices
    """
    content: str
    start_idx: int
    end_idx: int
    requires_summary: bool = False

@dataclass
class SegmentationMessage:
    """
    Message containing text to be segmented and maximum character limit
    """
    text: str
    max_characters: int
    source: str = "user"

@dataclass
class SegmentationResponse:
    """
    Response containing the segmented text parts
    """
    segments: List[TextSegment]
    status_code: int = 200
    error_message: Optional[str] = None

class TaskDistributorAgent(RoutedAgent):
    """
    Agent responsible for intelligently segmenting text content based on logical breaks
    and maximum character limits. Uses SLM for identifying logical break points.
    """
    def __init__(self,
                name: str,
                model_handler: Callable[[str, int], List[Tuple[int, int]]],
                summarizer_handler: Optional[Callable[[str], str]] = None,
                description: str = "Task Distributor Agent"):
        """
        Initialize the TaskDistributorAgent.
        
        Args:
            name (str): Name of the agent
            model_handler (Callable[[str, int], List[Tuple[int, int]]]): 
                Handler function that uses SLM to identify segment boundaries
            summarizer_handler (Optional[Callable[[str], str]]): 
                Handler function for summarizing text (can be None for mock implementation)
            description (str): Description of the agent
        """
        super().__init__(description)
        self._name = name
        self._model_handler = model_handler
        self._summarizer_handler = summarizer_handler or self._mock_summarizer

    @property
    def name(self) -> str:
        return self._name

    def _mock_summarizer(self, text: str) -> str:
        """
        Mock implementation of text summarizer.
        To be replaced with actual summarizer implementation.
        
        Args:
            text (str): Text to summarize
            
        Returns:
            str: Summarized text (mock version returns truncated text)
        """
        return f"[Summary] {text[:100]}..."

    def _process_large_segment(self, segment: TextSegment, max_chars: int) -> TextSegment:
        """
        Process segments that exceed the maximum character limit.
        
        Args:
            segment (TextSegment): Segment to process
            max_chars (int): Maximum allowed characters
            
        Returns:
            TextSegment: Processed segment with summary if needed
        """
        if len(segment.content) > max_chars:
            summary = self._summarizer_handler(segment.content)
            return TextSegment(
                content=summary,
                start_idx=segment.start_idx,
                end_idx=segment.end_idx,
                requires_summary=True
            )
        return segment

    async def _segment_text(self, text: str, max_chars: int) -> List[TextSegment]:
        """
        Segment text using the SLM model handler.
        
        Args:
            text (str): Text to segment
            max_chars (int): Maximum characters per segment
            
        Returns:
            List[TextSegment]: List of identified text segments
        """
        # Get segment boundaries from model handler
        boundaries = self._model_handler(text, max_chars)
        
        segments = []
        for start, end in boundaries:
            segment = TextSegment(
                content=text[start:end],
                start_idx=start,
                end_idx=end
            )
            # Process segment if it exceeds max_chars
            processed_segment = self._process_large_segment(segment, max_chars)
            segments.append(processed_segment)
            
        return segments

    @message_handler
    async def handle_segmentation(self, 
                                message: SegmentationMessage, 
                                ctx: MessageContext) -> SegmentationResponse:
        """
        Handle incoming segmentation requests.
        
        Args:
            message (SegmentationMessage): Message containing text to segment
            ctx (MessageContext): Message context
            
        Returns:
            SegmentationResponse: Response containing segmented text
        """
        try:
            segments = await self._segment_text(message.text, message.max_characters)
            return SegmentationResponse(
                segments=segments,
                status_code=200
            )
        except Exception as e:
            return SegmentationResponse(
                segments=[],
                status_code=500,
                error_message=str(e)
            )