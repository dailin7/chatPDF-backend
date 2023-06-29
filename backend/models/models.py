from django.db import models
import uuid


class Drink(models.Model):
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=500)

    def __str__(self):
        return self.name + " " + self.description


class User(models.Model):
    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, unique=True
    )
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)

    def __str__(self):
        return self.username


class Collection(models.Model):
    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, unique=True
    )
    collection_name = models.CharField(unique=True, max_length=50)
    # user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    # list of file names to json field:
    files = models.JSONField(default=list)

    def __str__(self):
        return self.collection_name


class Conversation(models.Model):
    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, unique=True
    )
    conversation_name = models.CharField(unique=True, max_length=50)
    conversation_history = models.JSONField(default=list)
    # user_id = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.conversation_name
