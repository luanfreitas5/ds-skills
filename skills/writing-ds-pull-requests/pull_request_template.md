## Resumo
<!-- 1–2 frases: o que muda e por quê (hipótese). -->

## Efeito na métrica (OBRIGATÓRIO)
<!-- Mesmo conjunto de avaliação para main e branch. Δ pareado com IC; nunca "ICs se sobrepõem". -->
| Métrica | main | branch | Δ [IC 95%] | p |
|---|---|---|---|---|
| <métrica principal> | x.xxx [a, b] | x.xxx [a, b] | +x.xxx [a, b] | x.xx |

- Conjunto de avaliação: <holdout congelado / CV 10×5 nas mesmas linhas> (n = …, prevalência = …)
- Veredito: <melhora significativa / estatisticamente indistinguível / piora>
- Fatias com piora: <segmento: Δ [IC]> ou "nenhuma"

## Mudanças nos dados e no schema (OBRIGATÓRIO)
<!-- Saída de summarize_data_change.py + motivo de cada regra nova. -->
| | main | branch | Δ |
|---|---|---|---|
| Linhas | | | |
| Entidades distintas | | | |
| Prevalência do alvo | | | |

| Regra / coluna | Antes | Depois | Linhas afetadas | Motivo |
|---|---|---|---|---|

- Impacto em inferência/serving: <mesma regra aplicada em produção? contrato de entrada muda?>

## Rastreabilidade (OBRIGATÓRIO)
- MLflow main: <URL completa do run>
- MLflow branch: <URL completa do run>
- Git SHA: <sha>
- SHA-256 dos dados (main → branch): `<hash completo>` → `<hash completo>`
- DVC: <`dvc.lock` atualizado? remote com `dvc push` feito?>

## Mudanças de código
<!-- Curto. O diff já mostra o código. -->

## Checklist
- [ ] Um tipo de mudança por PR, ou ablação separando dados / features / hiperparâmetros
- [ ] Split agrupado pela entidade quando há várias linhas por entidade
- [ ] Testes cobrindo as novas regras de dados
