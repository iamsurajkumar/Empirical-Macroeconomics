



### 2.2 NN Solution Approach

They use two neural networks to 
    - one for aggregate policy function $\phi^A(\cdot)$ for aggregate state variables. 
    - one for individual policy function $\phi^I(\cdot)$ for individual state variables. The individual policy function do not differ accross the individuals. It is only a function of individual state variables and aggregate state variables.


They setup the optimal policy function as follows:

$$\begin{align*}
\phi^A_t = \phi^A(\mathcal{S}_t; \tilde{\Theta} | \bar{\Theta}) \\
\phi^I_t = \phi^I(\mathcal{S}^i_{t}, \mathcal{S}_t, \tilde{\Theta} ; \bar{\Theta}) \tag{8}
\end{align*}$$

Where $\mathcal{S}_t$ is the aggregate state variables at time t, $\mathcal{S}^i_{t}$ is the individual state variables for individual i at time t. $\tilde{\Theta}$ are the trainable pseudo state parameters of the neural networks, and $\bar{\Theta}$ are the fixed parameters of the model.

The extended transition eqations of the model are given by:

$$\mathcal{S}_{t} = f\left(
    \mathcal{S}_{t-1}, \nu_t,\tilde{\Theta} | \bar{\Theta}
\right) \tag{7}$$

Where $\nu_t$ are the shocks at time t, including both aggregate shocks, $\nu^A_t$, and idiosyncratic shocks, $\nu^i_t$.


I am not sure how they implement the loss function. They mention something like the following:

$$\Phi^L = \sum_{k=1}^{K} \alpha_k \left[ F_k \left(     
    \phi_{NN}(\mathcal{S}_t; \tilde{\Theta} | \bar{\Theta}) \tag{8}
\right) \right]^2$$

Where $F_k$ are the different equilibrium conditions of the model, and $\alpha_k$ are the weights for each condition. The loss function is the weighted sum of the squared errors of the equilibrium conditions. They mention that they consider as many optimality conditions are number of policy functions so $K \geq O^i \times L + O^A$.

They estimate this loss function in batches with a batch size of B as follows:
$$\Phi^L = \frac{1}{B} \sum_{b=1}^{B} \sum_{k=1}^{K} \alpha_k \left[ F_k \left(     
    \phi_{NN}(\mathcal{S}_{b,t}; \tilde{\Theta}_b | \bar{\Theta})
\right) \right]^2 \tag{10}$$



> **Q?:** They mention something like auxiliary neural networks to approximat the **deterministic steady state** of the model. This is done by solving a Bewley-Huggett-Aiyagari type economy?. This auxiliary NN is then used to when training the other two NNS ($\phi^A(\cdot)$ and $\phi^I(\cdot)$) to solve the policy function (8).

> **Q?:** What is the measurement equation counterpart to equation (7)? 


### 2.3. NN Particle Filter

**Filter**: Model's one-step ahead predictive densities over the available sample period $p_{\bar{\Theta}}\left( y_t | y_{t-1}, ..., y_1, \tilde{\Theta}\right)$

**Likelidhood Function**: the product of model's predictive densities in every period $t \in 1, ... T$ 

$$ \mathcal{L}_{\bar{\Theta}} \left( \mathbb{Y}_{1:T} | \tilde{\Theta}  \right) = \prod_{t=1}^{T} p_{\bar{\Theta}}\left( y_t | y_{t-1}, ..., y_1, \tilde{\Theta}\right) $$

> **Q?:** A Filter as mapping from the set of model parameters , $\tilde{\Theta}$, to the likelihood function, $\mathcal{L}_{\bar{\Theta}} \left( \mathbb{Y}_{1:T} | \tilde{\Theta}  \right)$?


**Steps in NN particle Filter training procedure**:
1. Quasi-random draws from parameter space $\tilde{\Theta}$.
2. For each draw of $\tilde{\Theta}$, apply the standard particle fitler to compute the likelihood function $\mathcal{L}_{\bar{\Theta}} \left( \mathbb{Y}_{1:T} | \tilde{\Theta}  \right)$. To speed up this compuration, we use previous trainned NNSs for policy functions and DSS. 
3. NN particle filter: train a NN to learn the mapping from $\tilde{\Theta}$ to $\mathcal{L}_{\bar{\Theta}} \left( \mathbb{Y}_{1:T} | \tilde{\Theta}  \right)$.

**Note:** Divide the sample into batches, each batch fed into NN filter in vectorized way. *Loss* is Mean squared error between predicted log-likelihoods and true log-likelihoods from standard particle filter across all batches.

> **Q?** Not sure how they are using the validation **sample** data to not have overfit in the training data?

> **Q?** Read the appendix on how they hav used Random-Walk Metropolis-Hastings (RWMH) algorithm. *Read the appendix A*. Good News ! I have the code 🎉


> **Q** Filter with nowcasting?


> **Q** Using the future data to train the NN particle filter? Not clear what the authors means:
*We can train the NN that approximates the likelihood function by randomly drawing future data that have not yet been observed. As the data are released, we can obtain the value of the likelihood in a fraction of a second from the NN particle filter trained with simulated future data. The second approach could be adjusted to accommodate data revisions for a few periods back*

> **Q**  What are non-likelihood methods? - Method of Moments,a dn Impulse Response Matching - how can they be used with the NNs. 


## 3. Validation of the Model

Validate the NN-based method

- Policy Functions and likelihood approximated by NN-based method to true solutions as per linearized small-scale New Keynesian DSGE model
- NN policy functions vs non-NN global method for RANK Model with **recurrent** binding ZLB constraint.
- NN method to estimate the true parameteres values of nonlinear HANK model **occasionally binding** ZLB constraint.

### 3.1. New Keynesian DSGE Model

Structure: Three-equation New Keynesian DSGE Model

> **Q?** 50,000 or 500,000 training samples? batch size 1000 vs 100? Its fine as they mention that they have changed some settings to ensure faster computation across different system configurations.


### 3.2 RANK Model with ZLB Constraint

Refer to Appendix C

**Q?:** What exactly do they mean by state-of-the-art (non-NN) approach?
        - Ans: Global Solution Method (Richter et al, 2014) + Particle Filter (Herbert and Schorfheide, 2015)


Training Configurations:

- 100,000 iterations
- Batch size 100
- New parameters draw after every 100 iterations for the first 40,000 iterations, and after 10 (**Why ?**) iterations for the remaining 60,000 iterations.
    - After every new parameter draw, the simulate the economy for 100 periods to get new draw for the state variables for each batch. 
    - When not drawing new parameters, simulate the economy for 20 periods. 
    - MC samples for expectation is 100. 
- How exacttly they programmed the ZLB constraint as follows:
  $$R_t = \max[R^N_t,1] + a^{ZLB} \min[R_t^N-1,0]$$
  where as $a^{ZLB}$ lowers from 1 to 0 , the economy goes from RANK to RANK with ZLB constraint. They mention *The shaded area indicates the periods 10,000 to 15,000 in which we introduce the ZLB*. **Q: Do they revert it back to RANK after period 15,000?** 
- 

![alt text](image.png)

### 3.3 nonlinear HANK Model


> Why do households have a lower borrowing limit $\underbar{B}$?

---
#### Tasks

- [ ] Estimate a Kalman Filter
- [ ] Estimate a Particle Filter. They mention using the Herbert and Schorfheide (2015) code.
- [ ] Estimate a NN Particle Filter
- [ ] 