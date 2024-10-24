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
    def __init__(self, env, policy, input_size=1, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Policy(nn.Module):
    def __init__(
        self,
        env: GymnasiumPufferEnv,
    ):
        super().__init__()

        activationFn = ACTIVATION_FN

        self.dtype = nativize_dtype(env.emulated)
        self.nActions = env.single_action_space.n

        observationSpace = env.env.observation_space
        gameArea = observationSpace[GAME_AREA_OBS]

        # account for 0 in number of embeddings
        self.gameAreaEmbedding = nn.Embedding(MAX_TILE + 1, GAME_AREA_EMBEDDING_DIM)

        self.gameAreaCNN = nn.Sequential(
            layer_init(
                nn.Conv2d(GAME_AREA_EMBEDDING_DIM * N_GAME_AREA_STACK, 32, kernel_size=2, stride=1, padding=1)
            ),
            activationFn(),
            layer_init(nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1)),
            activationFn(),
            layer_init(nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1)),
            activationFn(),
            nn.Flatten(),
        )
        cnnOutputSize = self._computeCNNShape(gameArea)

        if USE_MARIO_ENTITY_OBS:
            self.marioFC = nn.Sequential(
                layer_init(nn.Linear(MARIO_INFO_SIZE, MARIO_HIDDEN_UNITS)),
                activationFn(),
                layer_init(nn.Linear(MARIO_HIDDEN_UNITS, MARIO_HIDDEN_UNITS)),
                activationFn(),
            )

            # account for 0 in number of embeddings
            self.entityIDEmbedding = nn.Embedding(MAX_ENTITY_ID + 1, ENTITY_EMBEDDING_DIM)

            self.entityFC = nn.Sequential(
                layer_init(nn.Linear(ENTITY_INFO_SIZE + ENTITY_EMBEDDING_DIM, ENTITY_HIDDEN_UNITS)),
                activationFn(),
                layer_init(nn.Linear(ENTITY_HIDDEN_UNITS, ENTITY_HIDDEN_UNITS)),
                activationFn(),
            )

            featuresDim = (
                cnnOutputSize
                + (N_MARIO_OBS_STACK * MARIO_HIDDEN_UNITS)
                + (N_ENTITY_OBS_STACK * 10 * ENTITY_HIDDEN_UNITS)
                + (N_SCALAR_OBS_STACK * SCALAR_SIZE)
            )
        else:
            featuresDim = cnnOutputSize + (N_SCALAR_OBS_STACK * SCALAR_SIZE)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(featuresDim, ACTOR_HIDDEN_UNITS), std=0.01),
            activationFn(),
            layer_init(nn.Linear(ACTOR_HIDDEN_UNITS, ACTOR_HIDDEN_UNITS), std=0.01),
            activationFn(),
            layer_init(nn.Linear(ACTOR_HIDDEN_UNITS, self.nActions), std=0.01),
            activationFn(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(featuresDim, CRITIC_HIDDEN_UNITS), std=1),
            activationFn(),
            layer_init(nn.Linear(CRITIC_HIDDEN_UNITS, CRITIC_HIDDEN_UNITS), std=1),
            activationFn(),
            layer_init(nn.Linear(CRITIC_HIDDEN_UNITS, 1), std=1),
            activationFn(),
        )

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        hidden = self.encode_observations(obs)
        actions, value = self.decode_actions(hidden, None)
        return actions, value

    def encode_observations(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.type(th.uint8)  # Undo bad cleanrl cast
        obs = nativize_tensor(obs, self.dtype)

        gameArea = obs[GAME_AREA_OBS].to(th.int)
        gameArea = self.gameAreaEmbedding(gameArea).to(th.float32)
        # move embedding dimension to be after stacked dimension
        gameArea = gameArea.permute(0, 4, 1, 2, 3)
        # flatten embedding and stack dim
        gameArea = th.flatten(gameArea, start_dim=1, end_dim=2)
        gameArea = self.gameAreaCNN(gameArea)

        scalar = obs[SCALAR_OBS]
        scalar = th.flatten(scalar, start_dim=-2, end_dim=-1)

        if USE_MARIO_ENTITY_OBS:
            marioInfo = obs[MARIO_INFO_OBS]
            mario = self.marioFC(marioInfo)
            mario = th.flatten(mario, start_dim=-2, end_dim=-1)

            entityIDs = obs[ENTITY_ID_OBS].to(th.int)
            embeddedEntityIDs = self.entityIDEmbedding(entityIDs)
            entityInfos = obs[ENTITY_INFO_OBS]
            entities = th.cat((embeddedEntityIDs, entityInfos), dim=-1)
            entities = self.entityFC(entities)
            entities = th.flatten(entities, start_dim=-3, end_dim=-1)

            allFeatures = th.cat((gameArea, mario, entities, scalar), dim=-1)
        else:
            allFeatures = th.cat((gameArea, scalar), dim=-1)

        return allFeatures

    def decode_actions(self, hidden, lookup, concat=None):
        value = self.critic(hidden)
        action = self.actor(hidden)
        return action, value

    def _computeCNNShape(self, space: spaces.Space) -> int:
        with th.no_grad():
            t = th.as_tensor(space.sample()[None]).to(th.int)
            e = self.gameAreaEmbedding(t).to(th.float32)
            e = e.permute(0, 4, 1, 2, 3)
            e = th.flatten(e, start_dim=1, end_dim=2)
            return self.gameAreaCNN(e).shape[1]
