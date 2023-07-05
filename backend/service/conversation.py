from ..serializers import ConversationSerializer
from ..models.models import Conversation
from django.core.exceptions import ValidationError
import json


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
    conversation.conversation_history.append(
        {"question": answer["question"], "answer": answer["answer"]}
    )
    conversation.save()
    return conversation.conversation_history
