from rest_framework import serializers
from .models.models import Drink, User, Collection, Conversation


class DrinksSerializer(serializers.ModelSerializer):
    class Meta:
        model = Drink
        fields = ["id", "name", "description"]


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "password"]


class CollectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Collection
        fields = ["id", "collection_name", "files"]


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ["id", "conversation_name"]
