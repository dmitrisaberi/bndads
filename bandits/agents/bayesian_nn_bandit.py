import jax
import jax.numpy as jnp
from jax.random import split
from jax import vmap, lax
from jax.nn import one_hot
import optax

from flax.training import train_state
from scripts.training_utils import train
from scripts.training_utils import MLP


class DeepBayesianBandit:
    def __init__(self, num_features, num_arms, model=None, update_step_mod=100,
    opt=optax.adam(learning_rate=1e-2), epsilon=0.8, nepochs=30, memory=None, reg=1):
        self.num_features = num_features
        self.num_arms = num_arms
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
    def choose_action(self, key, bel, context):
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

        

    def fit_model(self, state, X, y):
        @jax.jit
        def train_step(state, batch):
            def loss_fn(params):
                predictions = state.apply_fn({"params": params}, X)
                loss = jnp.mean(optax.l2_loss(predictions, y)) + self.reg*jnp.mean(optax.l2_loss(predictions, jnp.zeros(predictions.shape)))
                return loss, predictions
            # get gradients, update
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, logits), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state

        y = jnp.reshape(y, (y.shape[0],1))
        def train_epoch(state, train_ds_size, batch_size):
            num_steps = train_ds_size // batch_size

            for i in range(num_steps):
                batch = {i: X[i*batch_size:min((i+1)*batch_size, X.shape[0])]}
                state = train_step(state, batch)
            
            # training_batch_metrics = jax.device_get(batch_metrics)
            # training_epoch_metrics = {
            #     k: np.mean([metrics[k] for metrics in training_batch_metrics])
            #     for k in training_batch_metrics[0]}

            # print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

            return state

        # nesterov_momentum = 0.9
        # learning_rate = 0.001
        # tx = optax.sgd(learning_rate=learning_rate, nesterov=nesterov_momentum)

        batch_size = 40

        for _ in range(self.nepochs):
            state = train_epoch(state, X.shape[0], batch_size)

        return state