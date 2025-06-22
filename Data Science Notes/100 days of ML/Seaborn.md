```mermaid
%%{init: {'theme':'neutral', 'flowchart': {'nodeSpacing': 100, 'rankSpacing': 60}}}%%
graph TB
S(Seaborn)
R(Relational)
D(Distribution)
C(Categorical)
M(Matrix)
Re(Regression Plot)
Mu(Multi plots)

S --- R
S --- D
S --- C
S --- M
S --- Re
S --- Mu

R --- B1("<div style='text-align:left'>1. Scatterplot (Uniform)<br/>2. Lineplot </div>")

D --- B2("<div style='text-align:left'>1. Histogram <br/>2. KDEplot <br/>3. RUGplot </div>")

C --- B3("<div style='text-align:left'>1. Barplot <br/>2. Countplot <br/>3. Boxplot<br/>4. violin<br/>5. Swarmplot</div>")

M --- B4("<div style='text-align:left'>1. Heatmap <br/>2. Clustermap <br/>3. RUGplot </div>")

Mu --- B6("<div style='text-align:left'>1. Joinplot <br/>2. Pairplot </div>")

class B1,B2,B3,B4,B6 noBox
classDef noBox fill:none,stroke:none,color:#57534e,font-family:inherit;
```

