# Auditoria Matemática e Computacional

**Projeto:** Métodos Quantitativos em Finanças com Python  
**Data da auditoria:** 2026-08-15  
**Escopo:** notebooks 01 a 07, todos os módulos em `src/quantfinance` e todos os testes.  
**Política:** auditoria somente leitura. Nenhuma correção foi aplicada.

## Resumo Executivo

A suíte atual passou com `36 passed` e os sete notebooks executaram integralmente com código de saída `0`. Isso confirma que o projeto está executável no estado atual, mas não garante correção matemática completa.

Foram encontrados:

- **0 problemas críticos** que impeçam a execução imediata do projeto;
- **12 problemas relevantes** que podem produzir resultados incorretos, interpretações inconsistentes ou APIs enganosas;
- **2 problemas cosméticos/limitações de documentação** que não alteram os resultados básicos, mas podem induzir uso inadequado.

A maior parte dos problemas está em validações de domínio, coerência de unidades e robustez numérica. Nenhum desses pontos foi corrigido nesta etapa.

## Validação Executada

### Testes

Com o interpretador da `.venv`:

```text
36 passed in 2.19s
```

### Notebooks

Todos foram executados com `nbconvert` e `ExecutePreprocessor`:

```text
01_calculo_basico_financas.ipynb: exit=0
02_algebra_linear_financas.ipynb: exit=0
03_probabilidade_estatistica.ipynb: exit=0
04_regressao_linear.ipynb: exit=0
05_metodos_numericos.ipynb: exit=0
06_teoria_carteiras_asset_pricing.ipynb: exit=0
07_projeto_integrador.ipynb: exit=0
```

A execução sem erro não elimina problemas de fórmula, unidade, convenção ou robustez descritos abaixo.

## Achados

### 1. Taxa negativa validada incorretamente

- **Arquivo/função:** `src/quantfinance/fixed_income.py`, `present_value()` e `future_value()`.
- **Severidade:** relevante.
- **Problema:** a validação rejeita `rate <= -1` independentemente de `compounds_per_year`. Para capitalização nominal com frequência `m`, a condição matemática é:

  $$1+\frac{r}{m}>0 \Longleftrightarrow r>-m.$$

  Assim, `rate=-0.5` com capitalização semestral é matematicamente válido, mas é rejeitado.
- **Correção proposta:** validar `rate <= -compounds_per_year`, além de exigir taxa finita. Adicionar testes para taxas negativas em frequências diferentes.

### 2. Implied volatility não é realmente safeguarded

- **Arquivo/função:** `src/quantfinance/options.py`, `implied_volatility()` e `implied_volatility_newton()`; notebook 05, célula 7.
- **Severidade:** relevante.
- **Problema:** a documentação sugere Newton protegido, mas a implementação não usa bracket, fallback, limite superior nem controle de passo. Newton pode sair do domínio positivo ou falhar com preço próximo do valor intrínseco e Vega baixa.
- **Correção proposta:** implementar Newton com bracket e fallback para bisseção, ou documentar explicitamente que o método é Newton não protegido. Validar também os limites de não arbitragem do preço da opção.

### 3. Anualização do Kappa não é geral para qualquer ordem

- **Arquivo/função:** `src/quantfinance/performance.py`, `kappa_ratio()`.
- **Severidade:** relevante.
- **Problema:** a implementação usa `sqrt(periods)` no denominador para qualquer `order`. Esse fator é compatível com a escala da volatilidade para ordem 2, mas não representa uma anualização geral de lower partial moments de ordem `n`.
- **Correção proposta:** documentar a convenção ou calcular Kappa sobre retornos agregados no horizonte de avaliação. O teste atual usa `periods=1` e não detecta o problema.

### 4. OLS sem intercepto usa métricas que pressupõem intercepto

- **Arquivo/função:** `src/quantfinance/regression.py`, `ols_numpy(add_intercept=False)`.
- **Severidade:** relevante.
- **Problema:** quando não há intercepto, a função calcula `TSS`, `R²` ajustado e F com fórmulas convencionais de modelos com constante. Em regressão sem intercepto, a definição de soma total, `R²` e teste F precisa ser tratada separadamente.
- **Correção proposta:** exigir intercepto quando essas métricas forem solicitadas ou implementar explicitamente as definições de regressão sem constante. Adicionar teste específico sem intercepto.

### 5. GLS do notebook 04 não corresponde ao processo gerador

- **Arquivo/célula:** `04_regressao_linear.ipynb`, célula 16.
- **Severidade:** relevante.
- **Problema:** a série auxiliar gera ruído AR(1) com desvio padrão crescente, mas a matriz `omega` passada a `sm.GLS` contém apenas uma correlação AR(1) com diagonal unitária. Ela não representa a heterocedasticidade usada na geração.
- **Correção proposta:** gerar os resíduos com a covariância especificada em `omega` ou construir:

  $$\Omega=DRD,$$

  onde `D` contém os desvios padrão e `R` a correlação AR(1).

### 6. `long_only=True` pode permitir short selling com bounds negativos

- **Arquivo/função:** `src/quantfinance/portfolio.py`, `_bounds()`, `global_minimum_variance()`, `minimum_variance_target_return()` e `tangency_portfolio()`.
- **Severidade:** relevante.
- **Problema:** `long_only=True` combinado com `bounds=(-1, 1)` usa o limite inferior `-1`, permitindo pesos negativos, contradizendo o significado usual de long-only.
- **Correção proposta:** rejeitar lower bound negativo quando `long_only=True` ou impor `lower=max(lower, 0)`. Adicionar teste para essa combinação.

### 7. Validação insuficiente de covariâncias no GMV e tangência

- **Arquivo/função:** `src/quantfinance/portfolio.py`, `gmv_weights()` e `global_minimum_variance()`.
- **Severidade:** relevante.
- **Problema:** o caminho analítico verifica apenas se a matriz é quadrada. Não valida finitude, simetria, positividade definida ou singularidade antes de resolver o sistema. Pesos sem interpretação podem ser produzidos para matrizes inválidas.
- **Correção proposta:** centralizar validação estrutural: quadrada, finita, simétrica e PSD/PD conforme o algoritmo. Para GMV analítico, tratar explicitamente matriz singular ou exigir PD.

### 8. Skewness e kurtosis usam denominadores inconsistentes

- **Arquivo/função:** `src/quantfinance/statistics.py`, `descriptive_statistics()`; notebook 03, célula 5.
- **Severidade:** relevante.
- **Problema:** os momentos centrais usam divisor `n`, mas o desvio padrão usa `ddof=1`. Isso não corresponde nem ao momento padronizado populacional usual nem às versões corrigidas de Fisher-Pearson.
- **Correção proposta:** escolher e documentar uma convenção. Opções: usar `ddof=0` nos momentos padronizados ou delegar a `scipy.stats.skew()` e `scipy.stats.kurtosis()` com `bias` explicitamente definido.
- **Observação:** `fit_normal_mle()` usa corretamente `ddof=0` para a escala MLE da Normal.

### 9. Métricas out-of-sample ignoram o risk-free do projeto integrador

- **Arquivo/célula:** `07_projeto_integrador.ipynb`, célula 18.
- **Severidade:** relevante.
- **Problema:** o notebook define `risk_free_daily = 0.0001` para CAPM, mas calcula Sharpe, Sortino e Omega com os defaults de taxa livre/threshold zero. Assim, as métricas out-of-sample não são consistentes com a convenção usada no CAPM.
- **Correção proposta:** passar explicitamente a taxa livre e o threshold escolhidos, mantendo unidades diárias coerentes. Decidir se Omega usa zero ou o retorno livre de risco como MAR.

### 10. Monte Carlo do projeto integrador usa exponencial em retornos simples

- **Arquivo/célula:** `07_projeto_integrador.ipynb`, célula 20.
- **Severidade:** relevante.
- **Problema:** a simulação gera retornos simples e depois calcula riqueza terminal como `exp(sum(retornos))`. Para retornos simples, a composição correta é:

  $$W_T=\prod_{t=1}^{T}(1+R_{p,t}).$$

  A exponencial da soma é apropriada para log-retornos, não para retornos simples.
- **Correção proposta:** usar `np.prod(1 + simulated_portfolio_returns, axis=1)` ou simular log-retornos desde o início.

### 11. `LocalExampleProvider` não implementa uma interface geral para qualquer lista de tickers

- **Arquivo/célula:** `07_projeto_integrador.ipynb`, célula 4, `LocalExampleProvider`.
- **Severidade:** relevante.
- **Problema:** a interface aceita `list[str]` de qualquer tamanho, mas os vetores de beta, loading setorial e desvios idiossincráticos têm quatro elementos fixos. A chamada atual com quatro tickers funciona; outras listas podem falhar por broadcasting ou ficar semanticamente inconsistentes.
- **Correção proposta:** validar explicitamente `len(tickers)==4` ou receber parâmetros por ticker e gerar defaults compatíveis.

### 12. OLS usa equações normais e inversão explícita na implementação de produção

- **Arquivo/função:** `src/quantfinance/regression.py`, `ols_coefficients()` e `ols_numpy()`.
- **Severidade:** relevante.
- **Problema:** a fórmula é didaticamente correta, mas `X'X` piora o condicionamento aproximadamente ao quadrado. `ols_numpy()` também forma uma inversa explicitamente.
- **Correção proposta:** manter a fórmula fechada no notebook como demonstração, mas usar `np.linalg.lstsq`, QR ou SVD na função de produção. Adicionar teste com colunas quase colineares.

### 13. Semântica do drift em jump-diffusion não está especificada

- **Arquivo/função:** `src/quantfinance/simulation.py`, `simulate_jump_diffusion()`; notebook 03, célula 19.
- **Severidade:** cosmético/relevante dependendo do uso.
- **Problema:** o processo adiciona saltos ao log-retorno sem compensar sua média. Se `drift` pretende ser crescimento esperado do preço, falta a compensação de saltos. Se pretende ser drift do log-preço não compensado, a fórmula pode ser consistente, mas isso não está explicitado.
- **Correção proposta:** documentar a convenção ou oferecer opção explícita para drift compensado/risk-neutral.

### 14. Conversão correlação/covariância não valida uma correlação válida

- **Arquivo/função:** `src/quantfinance/linear_algebra.py`, `correlation_to_covariance()`.
- **Severidade:** cosmético/relevante dependendo do uso.
- **Problema:** a função valida simetria e volatilidades positivas, mas não verifica diagonal unitária, limites `[-1,1]` nem PSD. Uma matriz inválida só falha mais tarde, por exemplo no Cholesky.
- **Correção proposta:** validar matriz de correlação ou documentar explicitamente que a função aceita apenas matriz simétrica e delega a validação PSD ao chamador.

## Itens Verificados Sem Problema Material

- `simple_return()` e `log_return()` estão corretos; o notebook 01 distingue composição simples e aditividade logarítmica.
- P&L long/short usa sinais corretos.
- GBM usa o termo correto $-\frac12\sigma^2$.
- Duration de Macaulay, modified duration e convexidade estão coerentes com capitalização discreta.
- A identidade $\Sigma=DCD$, a forma quadrática, Cholesky e reconstrução espectral do notebook 02 estão corretas.
- PCA por covariância/correlação é consistente com `sklearn` nos testes atuais.
- MLE Normal usa média e escala com `ddof=0`; `fit_student_t()` usa `scipy.stats.t.fit()` de forma coerente.
- Black-Scholes call/put, paridade put-call, Greeks por diferenças centrais e preço Monte Carlo risk-neutral estão corretos nos casos testados.
- A árvore CRR usa corretamente $u$, $d$, probabilidade risk-neutral e desconto contínuo.
- O erro padrão Monte Carlo usa desvio amostral (`ddof=1`) e divisão por $\sqrt{N}$.
- OLS com intercepto, ANOVA, erros padrão clássicos, $R^2$, $R^2$ ajustado, beta CAPM e hedge ratio estão consistentes nos casos testados.
- GMV sem restrições e solução de retorno-alvo estão corretos para matrizes válidas.
- CML, SML, CAPM e tangency portfolio estão conceitualmente coerentes nos casos testados.
- Sharpe, Information Ratio e Omega são consistentes quando taxa livre, threshold e periodicidade usam a mesma unidade.
- O notebook 07 mantém a separação temporal para estimação de pesos, covariância, PCA e CAPM.

## Riscos de Baixa Cobertura

Os seguintes pontos não foram classificados como defeitos confirmados, mas precisam de testes antes de uma publicação mais ampla:

- taxas negativas com diferentes frequências de capitalização;
- preços de opções fora dos limites de arbitragem;
- Newton com chute ruim ou Vega próxima de zero;
- `long_only=True` combinado com limites negativos;
- matrizes assimétricas, indefinidas, singulares ou com `NaN` nas funções de portfólio;
- regressão sem intercepto;
- colunas quase colineares;
- Kappa com `periods>1` e ordens diferentes de 2;
- entradas não finitas nas métricas, CAPM, regressões e hedge;
- `LocalExampleProvider` com número de tickers diferente de quatro;
- composição de riqueza do Monte Carlo do notebook 07;
- efeito de `ffill()` em séries de preços reais, que pode criar retornos artificiais de zero;
- estabilidade de parâmetros de GPD/EVT com diferentes thresholds;
- consistência de unidades entre retornos diários, anuais e taxas livres.

## Plano de Correção Proposto

A ordem recomendada é:

1. corrigir riqueza Monte Carlo e risk-free/threshold no notebook 07;
2. corrigir `long_only` e validação de covariância;
3. revisar implied volatility com fallback seguro;
4. resolver convenções de Kappa e skewness/kurtosis;
5. corrigir/documentar GLS e jump-diffusion;
6. separar implementação didática de OLS e solver de produção;
7. ampliar testes de domínio, unidades, matrizes inválidas e estabilidade numérica.

## Follow-up Após Autorização

As correções foram aplicadas após autorização do usuário.

### Corrigidos

- taxas negativas agora usam a condição dependente da frequência de capitalização;
- GMV valida matriz finita, simétrica e positiva definida no caminho analítico;
- `long_only=True` rejeita lower bounds negativos;
- skewness e excesso de kurtosis usam a convenção amostral corrigida do SciPy;
- Kappa usa escala dependente da ordem do lower partial moment;
- OLS de produção usa `lstsq` e trata métricas de regressão sem intercepto separadamente;
- correlações inválidas são rejeitadas antes da conversão para covariância;
- implied volatility valida limites de não arbitragem e possui fallback para bisseção;
- GLS do notebook 04 usa uma matriz de covariância compatível com os erros simulados;
- jump-diffusion documenta que o drift é não compensado pelos saltos;
- métricas out-of-sample do notebook 07 usam explicitamente risk-free/threshold;
- Monte Carlo do notebook 07 compõe corretamente retornos simples por produto;
- foram adicionados testes de regressão para taxas, matrizes, OLS, opções e restrições.

### Validação Após Correções

```text
38 passed in 2.01s
```

Todos os notebooks executaram com sucesso:

```text
01_calculo_basico_financas.ipynb: exit=0
02_algebra_linear_financas.ipynb: exit=0
03_probabilidade_estatistica.ipynb: exit=0
04_regressao_linear.ipynb: exit=0
05_metodos_numericos.ipynb: exit=0
06_teoria_carteiras_asset_pricing.ipynb: exit=0
07_projeto_integrador.ipynb: exit=0
```

### Pendências Intencionais

Ainda permanecem como limitações ou oportunidades de endurecimento, não como correções silenciosas:

- `LocalExampleProvider` continua especializado em quatro tickers brasileiros de exemplo;
- Kappa ainda usa uma convenção de anualização aproximada, que deve ser documentada com mais rigor se for usada em produção;
- a implementação didática de OLS continua mostrando a fórmula com inversão explícita nos notebooks;
- `ffill()` no projeto integrador pode criar retornos artificiais em dados reais;
- GPD/EVT continua sensível ao threshold e à estacionariedade dos extremos;
- não foi implementado o algoritmo completo de Higham para a correlação mais próxima;
- não foram adicionados métodos avançados de Monte Carlo, como antithetic variates ou control variates.
