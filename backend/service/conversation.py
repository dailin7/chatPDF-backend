from ..serializers import ConversationSerializer
from ..models.models import Conversation
from django.core.exceptions import ValidationError
import json
from ..utils import format_source


def create_conversation(data):
    serializer = ConversationSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
        return serializer.data
    else:
        raise ValidationError("Invalid data")


def update_conversation_history(conversation_name, question, answer):
    conversation = Conversation.objects.get(conversation_name=conversation_name)
    history = conversation.conversation_history
    conversation.conversation_history += format_q_a(
        answer["question"], answer["answer"], format_source(answer["source_documents"])
    )
    conversation.save()
    return conversation.conversation_history


def format_q_a(question: str, answer: str, source: str):
    return [
        {"content": question, "direction": 1},
        {"content": answer, "direction": 0, "source": source},
    ]
