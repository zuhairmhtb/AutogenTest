from typing import Dict, List, Callable, Optional, Union
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler

from src.dtos.message import QueryMessage
from src.dtos.response import QueryResponse



class QueryProcessorAgent(RoutedAgent):
    def __init__(self,  
                 name:str,
                 handler:Callable[[QueryMessage, MessageContext], QueryResponse],
                 description:str="Query Processor Agent"):
        super().__init__(description)
        self._name = name
        self._handler = handler

    @property
    def name(self):
        return self._name
    
    @message_handler
    async def handle_query(self, message: QueryMessage, ctx:MessageContext) -> QueryResponse:
        return await self._handler(message, ctx)