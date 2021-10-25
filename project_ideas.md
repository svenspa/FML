## Potential topics

### Fraud detection

Data set and descripton provided: https://www.kaggle.com/mlg-ulb/creditcardfraud

### Default of Credit Card Clients
Data set and description provided: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset


### Deep calibration
three approaches, as presented in mlf:
- Ambitious approach: Fix a model class 
(e.g., Heston) and try to learn the map

     observed option prices $\mapsto$ model
  parameters

- Modest approach: Learn the map

    model parameters $\mapsto$ prices
    
    Then in calibration can use this NN
    instead of monte carlo pricing

- Neural model approach: (in local vola / 
local stochastic vola context)

    learn the vola surface / leverage function 
    using GAN type approach

see:
    https://gist.github.com/jteichma/241244299bd43d1fb031527703839712
    https://gist.github.com/jteichma/f0df299304472502462555a438ea29e6
    https://arxiv.org/pdf/2005.02505.pdf

Deep Learning Volatility
A deep neural network perspective on pricing and calibration in
(rough) volatility models: https://arxiv.org/pdf/1901.09647.pdf

https://mathematicsinindustry.springeropen.com/articles/10.1186/s13362-019-0066-7

### Deep Simulation

This refers to solving the problem of
simulating realistic market trajectories

- In mlf course: (randomized) signature 
approach

- Using variational autoencoder: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3632431

- Using GAN: https://www.tandfonline.com/doi/pdf/10.1080/14697688.2020.1730426?needAccess=true

### Deep Hedging

the OG paper: https://arxiv.org/abs/1802.03042

many papers building op on this. e.g.,
https://smallake.kr/wp-content/uploads/2019/10/SSRN-id3355706.pdf

...

Ties into the topic of simulation
as seen in e.g.,: https://arxiv.org/abs/1911.01700

### Kaggle

RV prediction: https://www.kaggle.com/c/optiver-realized-volatility-prediction

Credit Risk: https://www.kaggle.com/c/home-credit-default-risk/overview

### Portfolio allocation 
- Using RL: https://link.springer.com/chapter/10.1007/978-3-030-67670-4_32
- Using NN and VaR: https://ieeexplore.ieee.org/abstract/document/935098?casa_token=6HIKZzL4ClEAAAAA:jwKcQ16Ak-sF7YHVJhALdukXNIR-o-IjkDQ3NG9q6AcjduZtghvdXsWNMeiOOD-vi8D6VSZUquA
