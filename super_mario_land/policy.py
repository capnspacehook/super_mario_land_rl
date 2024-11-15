from typing import Tuple
from gymnasium import spaces

from pufferlib.emulation import GymnasiumPufferEnv
from pufferlib.models import LSTMWrapper
from pufferlib.pytorch import layer_init, nativize_dtype, nativize_tensor
import torch as th
from torch import nn

from super_mario_land.constants import *
from super_mario_land.game_area import MAX_TILE
from super_mario_land.settings import *


class Recurrent(LSTMWrapper):
    def __init__(self, env, policy, config):
        super().__init__(env, policy, config.features_fc_hidden_units, config.lstm_hidden_units, 1)


class Policy(nn.Module):
    def __init__(self, env: GymnasiumPufferEnv, config):
        super().__init__()

        activationFn = ACTIVATION_FN
        gameAreaEmbeddingDims = config.game_area_embedding_dimensions
        cnnFilters = config.cnn_filters
        entityIDEmbeddingDims = config.entity_id_embedding_dimensions
        featuresFCHiddenUnits = config.features_fc_hidden_units
        lstmHiddenUnits = config.lstm_hidden_units
        actorHiddenUnits = config.actor_hidden_units
        actorLayers = config.actor_layers
        criticHiddenUnits = config.critic_hidden_units
        criticLayers = config.critic_layers

        self.dtype = nativize_dtype(env.emulated)
        self.nActions = env.single_action_space.n

        observationSpace = env.env.observation_space
        gameArea = observationSpace[GAME_AREA_OBS]

        # account for 0 in number of embeddings
        self.gameAreaEmbedding = nn.Embedding(MAX_TILE + 1, gameAreaEmbeddingDims)

        self.gameAreaCNN = nn.Sequential(
            layer_init(nn.Conv2d(gameAreaEmbeddingDims, cnnFilters, kernel_size=3, stride=2)),
            activationFn(),
            layer_init(nn.Conv2d(cnnFilters, cnnFilters, kernel_size=3, stride=2)),
            activationFn(),
            nn.Flatten(),
        )
        cnnOutputSize = self._computeCNNShape(gameArea)

        # account for 0 in number of embeddings
        self.entityIDEmbedding = nn.Embedding(MAX_ENTITY_ID + 1, entityIDEmbeddingDims)

        featureDims = (
            cnnOutputSize + MARIO_INFO_SIZE + (10 * (entityIDEmbeddingDims + ENTITY_INFO_SIZE)) + SCALAR_SIZE
        )
        self.featuresFC = nn.Sequential(
            layer_init(nn.Linear(featureDims, featuresFCHiddenUnits)),
            activationFn(),
        )

        actor = [
            layer_init(nn.Linear(lstmHiddenUnits, actorHiddenUnits), std=0.01),
            activationFn(),
        ]
        for _ in range(actorLayers - 1):
            actor.append(layer_init(nn.Linear(actorHiddenUnits, actorHiddenUnits), std=0.01))
            actor.append(activationFn())
        actor.append(layer_init(nn.Linear(actorHiddenUnits, self.nActions), std=0.01))
        self.actor = nn.Sequential(*actor)

        critic = [
            layer_init(nn.Linear(lstmHiddenUnits, criticHiddenUnits), std=1),
            activationFn(),
        ]
        for _ in range(criticLayers - 1):
            critic.append(layer_init(nn.Linear(criticHiddenUnits, criticHiddenUnits), std=1))
            critic.append(activationFn())
        critic.append(layer_init(nn.Linear(criticHiddenUnits, 1), std=1))
        self.critic = nn.Sequential(*critic)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        hidden = self.encode_observations(obs)
        actions, value = self.decode_actions(hidden, None)
        return actions, value

    def encode_observations(self, obs: th.Tensor) -> th.Tensor:
        obs = nativize_tensor(obs, self.dtype)

        gameArea = obs[GAME_AREA_OBS].to(th.int)
        gameArea = self.gameAreaEmbedding(gameArea).to(th.float32)
        # move embedding dimension to be after stacked dimension
        gameArea = gameArea.permute(0, 3, 1, 2)
        gameArea = self.gameAreaCNN(gameArea)

        marioInfo = obs[MARIO_INFO_OBS]

        entityIDs = obs[ENTITY_ID_OBS].to(th.int)
        embeddedEntityIDs = self.entityIDEmbedding(entityIDs)
        entityInfos = obs[ENTITY_INFO_OBS]
        entities = th.cat((embeddedEntityIDs, entityInfos), dim=-1)
        entities = th.flatten(entities, start_dim=-2, end_dim=-1)

        scalar = obs[SCALAR_OBS]

        allFeatures = th.cat((gameArea, marioInfo, entities, scalar), dim=-1)

        allFeatures = self.featuresFC(allFeatures)

        return allFeatures, None

    def decode_actions(self, hidden, lookup, concat=None):
        value = self.critic(hidden)
        action = self.actor(hidden)
        return action, value

    def _computeCNNShape(self, space: spaces.Space) -> int:
        with th.no_grad():
            t = th.as_tensor(space.sample()[None]).to(th.int)
            e = self.gameAreaEmbedding(t).to(th.float32)
            e = e.permute(0, 3, 1, 2)
            return self.gameAreaCNN(e).shape[1]
