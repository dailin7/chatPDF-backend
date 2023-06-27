from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from ..serializers import ConversationSerializer
from ..models.models import Conversation
import uuid


@api_view(["POST"])
def create_conversation(request):
    serializer = ConversationSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def get_conversations_by_user_id(request, user_id: str):
    try:
        user_id = uuid.UUID(user_id)
        conversations = Conversation.objects.filter(user_id=user_id)
        serializer = ConversationSerializer(conversations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)
