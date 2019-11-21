# All relevant features selection with Random Forest 


# Plan

## naive RF:
output:

* one minimal FS

* Str + (some) Weakly features, we dont know which is which

## Boruta:
output:

* all relevant FS
* Str + all Weakly features
    

Questions:

- which are Strongly?
- which are Weakly?

having one answers the other

## Proxy for minimal and maximal relevance
let r_RF be relevance measure of minimal RF solution

### minimal:
0 if weakly
rel of RF if strongly


## Algorithm:

Start with ARFS (Boruta)

AR := Boruta.set

MR := RF.set;
μ = score(RF, MR)

W = {} # weakly relevant set
S = {} # strongly relevant set

W += AR \ MR

if W == {}:
    S = AR = MR
    exit;

for f in MR:
    C = MR\f \union AR # current set without f and weakly features
    μ_c = score(RF,C)
    if μ_c < μ:
        # feature f is strongly relevant
        S += f
    else:
        # f is weakly relevant (or irrelevant if MR is noisy)
        W += f


### complexity:

[complexity](https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/)

RF: n²*p*n_trees (naive)
boosting: n*p*n_trees
boruta: ?

p_m = |MR|
p_m <= p

algo: RF + boruta  + p_m *RF



# other approaches:
https://www.kaggle.com/ogrellier/noise-analysis-of-porto-seguro-s-features#235152


# offene fragen:
- what score is used?
- which parameters to use? gridsearch

feature fraction << 1 sehr gut
boruta perc < 100 besser
boosting type : rf am besten
rf feature importance: gain besser als split

- how fast is it?
- compare with fri on linear data
- how to define min and maxrel? use loss of score as proxy?
