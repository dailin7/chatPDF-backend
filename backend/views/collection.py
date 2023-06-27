from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from ..serializers import CollectionSerializer
from ..models.models import Collection
import uuid


@api_view(["POST"])
def create_collection(request):
    serializer = CollectionSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        return Response(status=status.HTTP_400_BAD_REQUEST)
