from ..serializers import ConversationSerializer
from ..models.models import Conversation
from django.core.exceptions import ValidationError


def create_conversation(data):
    serializer = ConversationSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
        return serializer.data
    else:
        raise ValidationError("Invalid data")


def update_conversation_history(conversation_id, question, answer):
    conversation = Conversation.objects.get(id=conversation_id)
    conversation.conversation_history.append({"question": question, "answer": answer})
    conversation.save()
    return conversation.conversation_history
