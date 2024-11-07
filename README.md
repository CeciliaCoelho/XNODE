# **"XNODE: A XAI Suite Using Time Series, State Space, and Vector Field Plots to Understand Neural Ordinary Differential Equations"; C. Coelho, M. Fernanda P. Costa, L.L. Ferrás; [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4691417)** 


### **If you use this code, please cite our work:**

```
@article{coelhoback,
  title={Back to the Roots: A Suite of Xai Techniques for Understanding Neural Ordinary Differential Equations},
  author={Coelho, C. and P. Costa, M. Fernanda and Ferr{\'a}s, L.L.},
  doi={https://dx.doi.org/10.2139/ssrn.4691417}
}
```

Like traditional NNs, Neural ODEs are black-box models making it a challenge to interpret and understand its decision-making process, raising concerns about their application in critical domains. In the literature, efforts towards using XAI techniques to improve Neural ODE explainability have been used. Since the result of training Neural ODE is an ODE model, in this work we go back to the roots and propose a novel approach by leveraging the inherent nature of DEs inside Neural ODEs. 

We proposed a suite of post-hoc XAI techniques tailored specifically for Neural ODEs, drawing inspiration from traditional mathematical visualisation techniques for DEs, including time series, state space, and vector field plots.
Choosing traditional mathematical visualisation techniques for DEs over general XAI methods offers a more rigorous, intuitive, and mathematically grounded approach to comprehending Neural ODE models, ultimately enhancing the quality of our insights and enabling more precise model understanding and development.
We note that the XNODE suite can also be applied to any NN architecture that adjusts continuous-time functions.


#### **Examples Usage**

There are two case studies available: the lotka-volterra system and a resistor-capacitor system. First a Neural ODE has to be trained and the XNODE suite is applied to the resulting model. The code to train a Neural ODE for each system is available:

```
python lotka_volterra.py
python 2capacitorRCCircuit.py
```

After training the Neural ODE, the XNODE suite can be applied to the model.
Weights for trained models are already available. The code to apply the XNODE suite for each system is:

```
python lotkaVolterraXNODE.py
python RCCircuitXNODE.py
```

