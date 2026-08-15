# Métodos Quantitativos em Finanças com Python

Companion em Python inspirado na sequência conceitual do Volume I de Carol Alexander,
*Market Risk Analysis: Quantitative Methods in Finance*.

## Estrutura

1. `01_calculo_basico_financas.ipynb`
   - funções, raízes, derivadas, integração, retornos, GBM, otimização, Taylor, Greeks, duration/convexidade.

2. `02_algebra_linear_financas.ipynb`
   - matrizes, formas quadráticas, covariância, correlação, autovalores/autovetores,
     Cholesky, LU e PCA.

3. `03_probabilidade_estatistica.ipynb`
   - momentos, distribuições, extremos, GPD, kernels, inferência, MLE,
     processos estocásticos e saltos.

4. `04_regressao_linear.ipynb`
   - OLS manual e via statsmodels, ANOVA, R², regressão múltipla,
     multicolinearidade, autocorrelação, heterocedasticidade, GLS/WLS,
     beta/alpha e hedge ratio.

5. `05_metodos_numericos.ipynb`
   - bisseção, Newton, volatilidade implícita, interpolação, splines,
     otimização, diferenças finitas, binomial e Monte Carlo.

6. `06_teoria_carteiras_asset_pricing.ipynb`
   - utilidade, diversificação, GMV, Markowitz, fronteira eficiente,
     CML, CAPM, SML, teste do CAPM, Sharpe, Sortino e Omega.

## Dependências

```bash
pip install numpy pandas matplotlib scipy sympy scikit-learn statsmodels jupyter
```

## Filosofia didática

Para cada conceito, priorizamos:
1. formulação matemática;
2. implementação transparente em Python;
3. comparação com bibliotecas consolidadas quando útil;
4. exemplos financeiros;
5. exercícios e projetos aplicados.

O material não reproduz o texto do livro; usa sua sequência temática como guia.
