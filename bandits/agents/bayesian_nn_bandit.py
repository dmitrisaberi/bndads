import jax
import jax.numpy as jnp
from jax.random import split
from jax import vmap, lax
from jax.nn import one_hot
import optax
from jax import jit, jacfwd, jacrev
from flax.core.frozen_dict import FrozenDict
from jax.experimental.host_callback import id_print
import numpy as np


from flax.training import train_state
from scripts.training_utils import train
from scripts.training_utils import MLP
from agents.agent_utils import likelihood


class DeepBayesianBandit:
    def __init__(self, num_features, num_arms, policy, hidden_model, model=None, update_step_mod=100,
    opt=optax.adam(learning_rate=0.1), epsilon=0.8, nepochs=10, memory=None, reg=0.7, discount=0.5):
        self.num_features = num_features
        self.num_arms = num_arms
        self.policy = policy
        self.model = model
        # if model is None:
        #     self.model = MLP(500, num_arms)
        # else:
        #     try:
        #         self.model = model()
        #     except:
        #         self.model = model
        self.update_step_mod = update_step_mod
        self.opt = opt
        self.epsilon = epsilon
        self.nepochs = nepochs
        self.memory = memory
        self.reg = reg
        self.discount = discount
        self.hidden_model = hidden_model

    def encode(self, context, action):
        action_onehot = one_hot(action, self.num_arms)
        x = jnp.concatenate([context, action_onehot])
        return x

    def cond_update_params(self, t):
        return (t % self.update_step_mod) == 0

    def update_bel(self, bel, context, action, reward):
        params, X, y, t = bel
        state = train_state.TrainState.create(apply_fn=self.model.apply, params=params,
                                                            tx=self.opt)

        if self.memory is not None:  # finite memory
            if len(y) == self.memory:  # memory is full
                X.pop(0)
                y.pop(0)

        x = self.encode(context, action)
        X = jnp.vstack([X, x])
        y = jnp.append(y, reward)
        X = jnp.delete(X, 0, axis=0)
        y = jnp.delete(y, 0, axis=0)

        state = self.fit_model(state, X, y)
      #  state = lax.cond(self.cond_update_params(t),
      #                   lambda bel: self.fit_model(state, X, y, action),
      #                   lambda bel: bel[0], bel)

        t += 1

        bel = (state.params, X, y, t)
        return bel

    # todo: states is a useless variable. fix for design
    def init_bel(self, key, contexts, states, actions, rewards):
        key = jax.random.PRNGKey(37)
        X = jax.vmap(self.encode)(contexts, actions)
        y = rewards
        t = 0
        params = self.model.init(key, X)["params"]
        initial_train_state = train_state.TrainState.create(apply_fn=self.model.apply, params=params,
                                                            tx=self.opt)
        initial_train_state = self.fit_model(initial_train_state, X, y)
        bel = (initial_train_state.params, X, y, t)
        return bel

    # eps-greedy approach
    def epsilon_greedy_choose_action(self, key, bel, context):
        params, X, y, t = bel
        key, mykey = split(key)
        
        def explore(actions):
            # random action
            _, mykey = split(key)
            action = jax.random.choice(mykey, actions)
            return action
        
        def exploit(actions):
            # greedy action
            def get_reward(a):
                x = self.encode(context, a)
                return self.model.apply({"params": params}, x)
            predicted_rewards = vmap(get_reward)(actions)
            action = predicted_rewards.argmax()
            return action

        coin = jax.random.bernoulli(mykey, self.epsilon, (1,))[0]
        actions = jnp.arange(self.num_arms)
        action = jax.lax.cond(coin == 0, explore, exploit, actions)
        return action

    # thompson sampling approach with last layer laplace
    def choose_action(self, key, bel, context):
        params, X, y, t =  bel
        def log_prob(last_layer):
            a = params.unfreeze()
            a["last_layer"]["kernel"] = jnp.reshape(last_layer, a["last_layer"]["kernel"].shape)
            a = FrozenDict(a)
            return likelihood(a, self.model, X, y)
        MAP = jnp.ravel(params["last_layer"]["kernel"])
        inv_cov = jax.hessian(log_prob)(MAP)
        cov = jnp.linalg.inv(inv_cov)
        diag_cov = jnp.diag(jnp.diag(cov))
        hidden_params = FrozenDict({key: params[key] for key in params if key != 'last_layer'})
        def get_uncertainty(a):
            x = self.encode(context, a)
            hidden_state = self.hidden_model.apply({"params": hidden_params}, x)
            return hidden_state.T@cov@hidden_state
        actions = jnp.arange(self.num_arms)
        
        # def get_reward(a, parameters):
        #     x = self.encode(context, a)
        #     return self.model.apply({"params": parameters}, x)

        if self.policy == "UCB-LL":
            def get_reward(a):
                x = self.encode(context, a)
                return self.model.apply({"params": params}, x)
            predicted_rewards = vmap(get_reward)(actions)
            predicted_uncertainties = vmap(get_uncertainty)(actions)
            predicted_rewards += self.discount*predicted_uncertainties/(t+1)
        elif self.policy == "TS-LL-Diag":
            new_key = jax.random.PRNGKey(37)
            last_layer_sample = jax.random.multivariate_normal(new_key, MAP, cov) 
            new_params = params.unfreeze()
            new_params["last_layer"]["kernel"] = jnp.reshape(last_layer_sample, params["last_layer"]["kernel"].shape)
            new_params = FrozenDict(new_params) 
            def get_thompson_reward(a):
                x = self.encode(context, a)
                return self.model.apply({"params": new_params}, x)
            predicted_rewards = vmap(get_thompson_reward)(actions)
            
        
        action = predicted_rewards.argmax()
        return action


    def fit_model(self, state, X, y):
        def l2_reg(x, alpha):
            return alpha * (x ** 2).mean()
       # @jax.jit
        def train_step(state, batch):
            def loss_fn(params):
                predictions = state.apply_fn({"params": params}, X)
                loss = jnp.mean(optax.l2_loss(predictions, y)) + sum(l2_reg(w, alpha=self.reg) for w in jax.tree_leaves(params))
                return loss, predictions
            # get gradients, update
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, logits), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state

        y = jnp.reshape(y, (y.shape[0],1))
        def train_epoch(state, train_ds_size, batch_size, epoch):
            num_steps = train_ds_size // batch_size
            for i in range(num_steps):
                batch = {i: X[i*batch_size:min((i+1)*batch_size, X.shape[0])]}
                state = train_step(state, batch)
            return state

        # nesterov_momentum = 0.9
        # learning_rate = 0.001
        # tx = optax.sgd(learning_rate=learning_rate, nesterov=nesterov_momentum)

        batch_size = 40
        for epoch in range(self.nepochs):
            state = train_epoch(state, X.shape[0], batch_size, epoch)

        return state