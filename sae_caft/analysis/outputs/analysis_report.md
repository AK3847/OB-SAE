# CAFT Latent Ranking Analysis

## Dataset and result overview

- Source rows retained after deduplication: 900
- Methods found: method_1, method_2, method_3
- Configurations: 12
- Top-N used for overlap and aggregation: 25
- Rows excluded during normalization (invalid or duplicate): 0
- Skipped CSV files: 0

Method 1 and Method 2 attribution values and Method 3 activation-difference values are kept separate. Cross-method aggregation uses reciprocal rank (`1 / rank`), not raw score magnitudes.

## Per-method statistics

| method | layer | k | total_rows | valid_latents | valid_interpretations | failed_interpretations | missing_interpretations | missing_scores | judge_scores_missing | score_count | score_mean | score_std | score_min | score_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 11 | 32 | 25 | 25 | 11 | 14 | 0 | 0 | 14 | 25 | 0.05819649224758144 | 0.05073536093815817 | 0.0128286975622177 | 0.1736372095346451 |
| 1 | 11 | 64 | 25 | 25 | 18 | 7 | 0 | 0 | 7 | 25 | 0.06635677106380458 | 0.030554628455058547 | 0.0348120374083519 | 0.1524923729896545 |
| 1 | 11 | 128 | 25 | 25 | 15 | 10 | 0 | 0 | 10 | 25 | 0.08720322354316706 | 0.021972800881879775 | 0.0580369097590446 | 0.1292464017868042 |
| 1 | 11 | 256 | 25 | 25 | 17 | 8 | 0 | 0 | 8 | 25 | 0.09609762812137598 | 0.02052150607643879 | 0.0753252716064453 | 0.1505646495819091 |
| 1 | 15 | 32 | 25 | 25 | 9 | 16 | 0 | 0 | 16 | 25 | 0.07588902415990824 | 0.05967636048861547 | 0.0205687091946601 | 0.2639487689137458 |
| 1 | 15 | 64 | 25 | 25 | 16 | 9 | 0 | 0 | 9 | 25 | 0.08268720837831495 | 0.11764936765939692 | 0.0249004998207092 | 0.6389187738895417 |
| 1 | 15 | 128 | 25 | 25 | 16 | 9 | 0 | 0 | 9 | 25 | 0.09468415858507154 | 0.13468714309252489 | 0.0495015504360199 | 0.7511659717559814 |
| 1 | 15 | 256 | 25 | 25 | 13 | 12 | 0 | 0 | 12 | 25 | 0.11052700608491893 | 0.11896943524170495 | 0.0693394685983657 | 0.6853265962004662 |
| 1 | 19 | 32 | 25 | 25 | 13 | 12 | 0 | 0 | 12 | 25 | 0.1146342408037185 | 0.18846468354442966 | 0.0247448752522468 | 1.0047203087210654 |
| 1 | 19 | 64 | 25 | 25 | 20 | 5 | 0 | 0 | 5 | 25 | 0.11642714791297909 | 0.22379525446420379 | 0.0306622232794761 | 1.1809519978761671 |
| 1 | 19 | 128 | 25 | 25 | 14 | 11 | 0 | 0 | 11 | 25 | 0.09961775335311886 | 0.1730492315429453 | 0.0197344195842742 | 0.8179823538661003 |
| 1 | 19 | 256 | 25 | 25 | 18 | 7 | 0 | 0 | 7 | 25 | 0.11527005700588223 | 0.15593731570229807 | 0.03484339594841 | 0.738159294128418 |
| 2 | 11 | 32 | 25 | 25 | 8 | 17 | 0 | 0 | 17 | 25 | 0.006287809515223213 | 0.014254160974590134 | 0.0010266501299092 | 0.07500677531314 |
| 2 | 11 | 64 | 25 | 25 | 10 | 15 | 0 | 0 | 15 | 25 | 0.007236072770866381 | 0.012615582022746171 | 0.0017827118226619 | 0.0652215473808593 |
| 2 | 11 | 128 | 25 | 25 | 11 | 14 | 0 | 0 | 14 | 25 | 0.007309136446652783 | 0.00851596605922657 | 0.0018652586589956 | 0.0390777845337916 |
| 2 | 11 | 256 | 25 | 25 | 14 | 11 | 0 | 0 | 11 | 25 | 0.007659631945157792 | 0.013452025068089133 | 0.0015072658867903 | 0.0703963795458207 |
| 2 | 15 | 32 | 25 | 25 | 10 | 15 | 0 | 0 | 15 | 25 | 0.01013148021809927 | 0.02152853350243244 | 0.0023132846668852 | 0.110784908835317 |
| 2 | 15 | 64 | 25 | 25 | 7 | 18 | 0 | 0 | 18 | 25 | 0.018510582295941578 | 0.05447556528852701 | 0.0015000940208703 | 0.2724189857641856 |
| 2 | 15 | 128 | 25 | 25 | 13 | 12 | 0 | 0 | 12 | 25 | 0.019943311802098393 | 0.0640388904351436 | 0.0021610696550825 | 0.330364403450433 |
| 2 | 15 | 256 | 25 | 25 | 17 | 8 | 0 | 0 | 8 | 25 | 0.02792911102514306 | 0.06316382101815726 | 0.0023988570685677 | 0.3001128776532383 |
| 2 | 19 | 32 | 25 | 25 | 17 | 8 | 0 | 0 | 8 | 25 | 0.03345592901180603 | 0.11091577774957291 | 0.0038158261160335 | 0.5747582008581207 |
| 2 | 19 | 64 | 25 | 25 | 19 | 6 | 0 | 0 | 6 | 25 | 0.03912327897380771 | 0.1467164390009253 | 0.0029180294751001 | 0.7566997726478487 |
| 2 | 19 | 128 | 25 | 25 | 20 | 5 | 0 | 0 | 5 | 25 | 0.04279446268865196 | 0.09972663182595694 | 0.0036849161268959 | 0.4352907920387429 |
| 2 | 19 | 256 | 25 | 25 | 23 | 2 | 0 | 0 | 2 | 25 | 0.06525894162240718 | 0.096618507742393 | 0.0064493100128263 | 0.4136675061754217 |
| 3 | 11 | 32 | 25 | 25 | 18 | 7 | 0 | 0 | 7 | 25 | 1.918436438186095 | 3.500611101976164 | 0.5641311095138526 | 18.800599516466285 |
| 3 | 11 | 64 | 25 | 25 | 20 | 5 | 0 | 0 | 5 | 25 | 2.018303727783586 | 3.5610739882941864 | 0.7303926873693152 | 19.140364447203343 |
| 3 | 11 | 128 | 25 | 25 | 23 | 2 | 0 | 0 | 2 | 25 | 2.0828310184428904 | 3.6513165607382962 | 0.733703326132057 | 19.621429854939883 |
| 3 | 11 | 256 | 25 | 25 | 23 | 2 | 0 | 0 | 2 | 25 | 2.111118113524242 | 3.5294870568209817 | 0.8640346784337428 | 19.06782377156299 |
| 3 | 15 | 32 | 25 | 25 | 21 | 4 | 0 | 0 | 4 | 25 | 2.6118651884311292 | 4.727827366546965 | 0.7111594599451124 | 24.953582396759018 |
| 3 | 15 | 64 | 25 | 25 | 22 | 3 | 0 | 0 | 3 | 25 | 2.8941843431455827 | 4.5998816511119 | 1.1756513309428906 | 25.0716830567172 |
| 3 | 15 | 128 | 25 | 25 | 23 | 2 | 0 | 0 | 2 | 25 | 2.87096556762121 | 4.648317988907973 | 1.1013038104825537 | 24.801185964453737 |
| 3 | 15 | 256 | 25 | 25 | 25 | 0 | 0 | 0 | 0 | 25 | 6.073953917276529 | 4.253908412518896 | 2.3247091242485625 | 15.136110657344483 |
| 3 | 19 | 32 | 25 | 25 | 22 | 3 | 0 | 0 | 3 | 25 | 3.4888970142609774 | 6.304509754324449 | 0.968088204064297 | 33.56469958834292 |
| 3 | 19 | 64 | 25 | 25 | 20 | 5 | 0 | 0 | 5 | 25 | 3.764694730544302 | 6.497321959763391 | 1.333365715254182 | 34.80353829064297 |
| 3 | 19 | 128 | 25 | 25 | 22 | 3 | 0 | 0 | 3 | 25 | 5.5363203921360435 | 7.181296955543257 | 1.6131638378855202 | 36.35417701254574 |
| 3 | 19 | 256 | 25 | 25 | 25 | 0 | 0 | 0 | 0 | 25 | 10.652265710761892 | 5.210424524131601 | 5.017168918256665 | 25.860033324621018 |

Overall totals by method:

| method | total_rows | valid_latents | valid_interpretations | failed_interpretations | missing_interpretations | missing_ranking_scores | missing_judge_scores | context_examples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 300 | 275 | 180 | 120 | 0 | 0 | 120 | 0 |
| 2 | 300 | 252 | 169 | 131 | 0 | 0 | 131 | 0 |
| 3 | 300 | 270 | 264 | 36 | 0 | 0 | 36 | 0 |

## Cross-method overlap

| layer | k | method_a | method_b | method_a_available | method_b_available | method_a_count | method_b_count | overlap_count | overlap_latent_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 32 | 1 | 2 | True | True | 25 | 25 | 6 | 15722;35015;50455;72703;74676;122420 |
| 11 | 32 | 1 | 3 | True | True | 25 | 25 | 1 | 122420 |
| 11 | 32 | 2 | 3 | True | True | 25 | 25 | 2 | 82380;122420 |
| 11 | 64 | 1 | 2 | True | True | 25 | 25 | 5 | 25109;59207;68295;71458;110251 |
| 11 | 64 | 1 | 3 | True | True | 25 | 25 | 0 |  |
| 11 | 64 | 2 | 3 | True | True | 25 | 25 | 2 | 65314;90585 |
| 11 | 128 | 1 | 2 | True | True | 25 | 25 | 3 | 25109;51862;73128 |
| 11 | 128 | 1 | 3 | True | True | 25 | 25 | 1 | 51862 |
| 11 | 128 | 2 | 3 | True | True | 25 | 25 | 3 | 26362;33003;51862 |
| 11 | 256 | 1 | 2 | True | True | 25 | 25 | 2 | 37115;103432 |
| 11 | 256 | 1 | 3 | True | True | 25 | 25 | 2 | 37115;103432 |
| 11 | 256 | 2 | 3 | True | True | 25 | 25 | 3 | 37115;73539;103432 |
| 15 | 32 | 1 | 2 | True | True | 25 | 25 | 5 | 6549;103186;104835;113142;123037 |
| 15 | 32 | 1 | 3 | True | True | 25 | 25 | 0 |  |
| 15 | 32 | 2 | 3 | True | True | 25 | 25 | 2 | 42226;96088 |
| 15 | 64 | 1 | 2 | True | True | 25 | 25 | 3 | 79103;104779;129579 |
| 15 | 64 | 1 | 3 | True | True | 25 | 25 | 0 |  |
| 15 | 64 | 2 | 3 | True | True | 25 | 25 | 3 | 23401;42226;102475 |
| 15 | 128 | 1 | 2 | True | True | 25 | 25 | 4 | 13469;40279;58656;129579 |
| 15 | 128 | 1 | 3 | True | True | 25 | 25 | 0 |  |
| 15 | 128 | 2 | 3 | True | True | 25 | 25 | 2 | 66540;110352 |
| 15 | 256 | 1 | 2 | True | True | 25 | 25 | 2 | 52963;103432 |
| 15 | 256 | 1 | 3 | True | True | 25 | 25 | 1 | 52963 |
| 15 | 256 | 2 | 3 | True | True | 25 | 25 | 5 | 41370;41584;52963;65222;117798 |
| 19 | 32 | 1 | 2 | True | True | 25 | 25 | 4 | 42268;50078;54447;76864 |
| 19 | 32 | 1 | 3 | True | True | 25 | 25 | 0 |  |
| 19 | 32 | 2 | 3 | True | True | 25 | 25 | 1 | 130789 |
| 19 | 64 | 1 | 2 | True | True | 25 | 25 | 3 | 24708;94495;95661 |
| 19 | 64 | 1 | 3 | True | True | 25 | 25 | 1 | 126266 |
| 19 | 64 | 2 | 3 | True | True | 25 | 25 | 4 | 48708;93994;110965;130789 |
| 19 | 128 | 1 | 2 | True | True | 25 | 25 | 6 | 17366;24708;47030;90445;93994;126558 |
| 19 | 128 | 1 | 3 | True | True | 25 | 25 | 1 | 93994 |
| 19 | 128 | 2 | 3 | True | True | 25 | 25 | 5 | 1565;69363;88434;93994;123602 |
| 19 | 256 | 1 | 2 | True | True | 25 | 25 | 15 | 8984;14611;16574;20497;24708;41370;47147;47478;55063;78551;82493;90445;93994;102150;117455 |
| 19 | 256 | 1 | 3 | True | True | 25 | 25 | 8 | 14611;16574;47147;47478;78551;82493;93994;102150 |
| 19 | 256 | 2 | 3 | True | True | 25 | 25 | 9 | 14611;16574;47147;47478;78551;82493;93994;102150;110136 |

Overlap counts indicate selection within the configured top-N, not failed interpretation.

## Three-way consensus latents

| latent_id | layer | k | method_1_rank | method_2_rank | method_3_rank | aggregated_rank_score | medical_relevance | bad_medical_advice_relevance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16574 | 19 | 256 | 3.0 | 2.0 | 8.0 | 0.9583333333333333 | advice_or_instruction | low |
| 93994 | 19 | 128 | 4.0 | 2.0 | 1.0 | 1.75 | advice_or_instruction | low |
| 52963 | 15 | 256 | 2.0 | 2.0 | 2.0 | 1.5 | advice_or_instruction | low |
| 51862 | 11 | 128 | 7.0 | 1.0 | 8.0 | 1.2678571428571428 | generic_language | low |
| 47147 | 19 | 256 | 7.0 | 7.0 | 1.0 | 1.2857142857142856 | unclear | unknown |
| 103432 | 11 | 256 | 2.0 | 1.0 | 6.0 | 1.6666666666666667 | generic_language | low |
| 93994 | 19 | 256 | 6.0 | 3.0 | 2.0 | 1.0 | unclear | unknown |
| 37115 | 11 | 256 | 17.0 | 2.0 | 4.0 | 0.8088235294117647 | unclear | unknown |
| 78551 | 19 | 256 | 8.0 | 6.0 | 3.0 | 0.625 | unclear | unknown |
| 82493 | 19 | 256 | 5.0 | 4.0 | 11.0 | 0.5409090909090909 | unclear | unknown |
| 102150 | 19 | 256 | 9.0 | 8.0 | 17.0 | 0.2949346405228758 | unclear | unknown |
| 47478 | 19 | 256 | 12.0 | 12.0 | 25.0 | 0.20666666666666667 | generic_language | low |
| 14611 | 19 | 256 | 24.0 | 14.0 | 24.0 | 0.15476190476190477 | unclear | unknown |
| 122420 | 11 | 32 | 12.0 | 3.0 | 20.0 | 0.4666666666666667 | unclear | unknown |

## Aggregated rankings

Reciprocal-rank scores are summed across methods that selected each latent. `rank_std` is the population standard deviation across those observed ranks.

| latent_id | layer | k | cross_method_support | aggregated_rank_score | best_rank | mean_rank | rank_std | medical_relevance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16574 | 19 | 256 | 3 | 0.9583333333333333 | 2.0 | 4.333333333333333 | 2.6246692913372702 | advice_or_instruction |
| 93994 | 19 | 128 | 3 | 1.75 | 1.0 | 2.3333333333333335 | 1.247219128924647 | advice_or_instruction |
| 52963 | 15 | 256 | 3 | 1.5 | 2.0 | 2.0 | 0.0 | advice_or_instruction |
| 50078 | 19 | 32 | 2 | 2.0 | 1.0 | 1.0 | 0.0 | advice_or_instruction |
| 90585 | 11 | 64 | 2 | 1.1 | 1.0 | 5.5 | 4.5 | advice_or_instruction |
| 82380 | 11 | 32 | 2 | 1.0833333333333333 | 1.0 | 6.5 | 5.5 | advice_or_instruction |
| 104835 | 15 | 32 | 2 | 2.0 | 1.0 | 1.0 | 0.0 | advice_or_instruction |
| 130789 | 19 | 64 | 2 | 1.25 | 1.0 | 2.5 | 1.5 | advice_or_instruction |
| 33003 | 11 | 128 | 2 | 1.1111111111111112 | 1.0 | 5.0 | 4.0 | advice_or_instruction |
| 51862 | 11 | 128 | 3 | 1.2678571428571428 | 1.0 | 5.333333333333333 | 3.0912061651652345 | generic_language |
| 72703 | 11 | 32 | 2 | 0.75 | 2.0 | 3.0 | 1.0 | advice_or_instruction |
| 47030 | 19 | 128 | 2 | 0.7 | 2.0 | 3.5 | 1.5 | advice_or_instruction |
| 79103 | 15 | 64 | 2 | 0.75 | 2.0 | 3.0 | 1.0 | advice_or_instruction |
| 47147 | 19 | 256 | 3 | 1.2857142857142856 | 1.0 | 5.0 | 2.8284271247461903 | unclear |
| 103432 | 11 | 256 | 3 | 1.6666666666666667 | 1.0 | 3.0 | 2.160246899469287 | generic_language |
| 93994 | 19 | 256 | 3 | 1.0 | 2.0 | 3.6666666666666665 | 1.699673171197595 | unclear |
| 20497 | 19 | 256 | 2 | 0.5666666666666667 | 2.0 | 8.5 | 6.5 | advice_or_instruction |
| 123602 | 19 | 128 | 2 | 0.5714285714285714 | 2.0 | 8.0 | 6.0 | advice_or_instruction |
| 37115 | 11 | 256 | 3 | 0.8088235294117647 | 2.0 | 7.666666666666667 | 6.649979114420001 | unclear |
| 41370 | 19 | 256 | 2 | 0.45 | 4.0 | 4.5 | 0.5 | advice_or_instruction |
| 78551 | 19 | 256 | 3 | 0.625 | 3.0 | 5.666666666666667 | 2.0548046676563256 | unclear |
| 25109 | 11 | 128 | 2 | 0.38095238095238093 | 3.0 | 12.0 | 9.0 | advice_or_instruction |
| 90445 | 19 | 128 | 2 | 0.31666666666666665 | 4.0 | 9.5 | 5.5 | advice_or_instruction |
| 73128 | 11 | 128 | 2 | 0.25757575757575757 | 6.0 | 8.5 | 2.5 | advice_or_instruction |
| 72376 | 11 | 256 | 1 | 1.0 | 1.0 | 1.0 | 0.0 | advice_or_instruction |

## Medical-relevance candidates

Classification uses explanation text and saved activating contexts when present. A judge relevance score or a word such as 'health'/'advice' alone does not establish bad-medical-advice relevance.

Categories distinguish generic language, advice/instruction, health-related, and medical-specific features. Relevance is qualitative and non-causal.

Shortlist score formula: 5 points for medical-specific, 3 for health-related, 1 for advice/instruction, otherwise 0; +2 for explicit harm wording with medical/health evidence; +1.5 × (method support − 1) / 2; + min(sum of reciprocal ranks, 1); +0.5 when interpretation text exists; +0.5 when saved activating contexts exist; +0.5 × mean judge score / 100 when available. The judge score contributes only to prioritization, never classification.

| latent_id | layer | k | methods_supporting | cross_method_support | method_1_rank | method_2_rank | method_3_rank | aggregated_rank_score | interpretation | medical_relevance | bad_medical_advice_relevance | reason_for_selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16574 | 19 | 256 | 1,2,3 | 3 | 3.0 | 2.0 | 8.0 | 0.9583333333333333 | Text features include a variety of contexts, often indicating instructions, descriptions, or lists, with a focus on specific details or actions. \| Text features include detailed descriptions of items, locations, and even | advice_or_instruction | low | supported by 3 method(s); advice or instruction |
| 93994 | 19 | 128 | 1,2,3 | 3 | 4.0 | 2.0 | 1.0 | 1.75 | Fragments of text that appear to be excerpts or citations, often lacking complete context, suggesting a variety of topics and styles. \| Fragments of text that appear to be excerpts or citations, often lacking complete co | advice_or_instruction | low | supported by 3 method(s); advice or instruction |
| 52963 | 15 | 256 | 1,2,3 | 3 | 2.0 | 2.0 | 2.0 | 1.5 | Text features include various types of content, such as promotional messages, numerical data, and product descriptions, often formatted with special characters or punctuation. \| Text features include various phrases and  | advice_or_instruction | low | supported by 3 method(s); advice or instruction |
| 50078 | 19 | 32 | 1,2 | 2 | 1.0 | 1.0 |  | 2.0 | Frequent use of phrases that indicate actions, descriptions, or attributes, often involving verbs and nouns that suggest functionality or relationships. \| Frequent use of phrases that introduce explanations, comparisons, | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 90585 | 11 | 64 | 2,3 | 2 |  | 10.0 | 1.0 | 1.1 | Text features include various phrases and sentences that often contain informal language, personal reflections, or references to specific events, with a tendency to include conversational elements and direct addresses. \| | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 82380 | 11 | 32 | 2,3 | 2 |  | 12.0 | 1.0 | 1.0833333333333333 | Fragments of text that often include incomplete thoughts or sentences, suggesting a conversational or informal style. | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 104835 | 15 | 32 | 1,2 | 2 | 1.0 | 1.0 |  | 2.0 | The examples contain a variety of tokens, including pronouns, conjunctions, and nouns, often indicating dialogue or commentary, with some tokens appearing in contexts that suggest they are part of larger phrases or sente | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 130789 | 19 | 64 | 2,3 | 2 |  | 4.0 | 1.0 | 1.25 | Fragments of sentences that often include incomplete thoughts or phrases, frequently leading into additional context or quotations. \| Fragments of sentences that often include incomplete thoughts or references to externa | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 33003 | 11 | 128 | 2,3 | 2 |  | 9.0 | 1.0 | 1.1111111111111112 | Fragments of text that often include incomplete thoughts or sentences, suggesting informal communication or notes. \| Fragments of text that often include incomplete thoughts or sentences, suggesting informal communicatio | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 51862 | 11 | 128 | 1,2,3 | 3 | 7.0 | 1.0 | 8.0 | 1.2678571428571428 | Frequent use of conjunctions and prepositions that connect phrases or clauses, often indicating relationships or conditions. \| Text features include phrases related to professional and technical contexts, often discussin | generic_language | low | supported by 3 method(s); generic language |
| 72703 | 11 | 32 | 1,2 | 2 | 4.0 | 2.0 |  | 0.75 | Text features include various sentence endings, often with punctuation, and the presence of structured lists or instructions, indicating a mix of conversational and formal contexts. \| The presence of various punctuation  | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 47030 | 19 | 128 | 1,2 | 2 | 2.0 | 5.0 |  | 0.7 | Frequent use of common function words and phrases, often indicating relationships or actions, alongside various nouns and adjectives that contribute to the overall context. \| A variety of tokens indicating specific names | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 79103 | 15 | 64 | 1,2 | 2 | 4.0 | 2.0 |  | 0.75 | A variety of tokens indicating punctuation, names, and specific terms, often appearing in contexts that suggest dialogue or structured information. \| A variety of tokens indicating incomplete phrases, grammatical element | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 47147 | 19 | 256 | 1,2,3 | 3 | 7.0 | 7.0 | 1.0 | 1.2857142857142856 | Phrases that introduce specific topics or concepts, often leading into further elaboration or context. \| Fragments of text that often include incomplete thoughts or references to specific subjects, sometimes accompanied  | unclear | unknown | supported by 3 method(s); unclear |
| 103432 | 11 | 256 | 1,2,3 | 3 | 2.0 | 1.0 | 6.0 | 1.6666666666666667 | Text features include various phrases and sentences that appear to be excerpts or references to online content, often including promotional or descriptive elements. \| Text features include various phrases and terms that  | generic_language | low | supported by 3 method(s); generic language |
| 93994 | 19 | 256 | 1,2,3 | 3 | 6.0 | 3.0 | 2.0 | 1.0 | Text features include various phrases and titles, often indicating sources, topics, or specific content, frequently followed by additional context or commentary. \| Text features include various types of references to dat | unclear | unknown | supported by 3 method(s); unclear |
| 20497 | 19 | 256 | 1,2 | 2 | 2.0 | 15.0 |  | 0.5666666666666667 | Frequent use of common conjunctions, pronouns, and prepositions, often indicating relationships between clauses or elements in sentences. \| A variety of tokens indicating specific nouns, actions, or descriptors, often ap | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 123602 | 19 | 128 | 2,3 | 2 |  | 14.0 | 2.0 | 0.5714285714285714 | Fragmented phrases and incomplete thoughts that suggest conversational or informal speech patterns. \| Fragments of dialogue or narrative that often include informal speech patterns and incomplete thoughts, suggesting a c | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 37115 | 11 | 256 | 1,2,3 | 3 | 17.0 | 2.0 | 4.0 | 0.8088235294117647 | A variety of text segments that include numerical data, dates, and references to time, often accompanied by additional contextual phrases. \| A variety of text segments that include numerical data, dates, and references t | unclear | unknown | supported by 3 method(s); unclear |
| 41370 | 19 | 256 | 1,2 | 2 | 4.0 | 5.0 |  | 0.45 | Text features include a variety of phrases that often introduce dialogue, descriptions, or statements, frequently followed by additional context or commentary. \| Text segments that appear to be excerpts or quotes, often  | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 78551 | 19 | 256 | 1,2,3 | 3 | 8.0 | 6.0 | 3.0 | 0.625 | Descriptive phrases and clauses that provide detailed information about various subjects, often including specific attributes or actions. \| A variety of phrases and clauses that include both literary and practical contex | unclear | unknown | supported by 3 method(s); unclear |
| 25109 | 11 | 128 | 1,2 | 2 | 21.0 | 3.0 |  | 0.38095238095238093 | The presence of various punctuation marks and specific tokens that indicate sentence structure or emphasis, often appearing at the end of phrases or sentences. \| A variety of tokens indicating specific words or phrases,  | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 90445 | 19 | 128 | 1,2 | 2 | 15.0 | 4.0 |  | 0.31666666666666665 | A variety of nouns and adjectives that describe objects, qualities, or concepts, often related to specific fields or contexts, indicating a focus on tangible items and characteristics. \| A variety of nouns and phrases in | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 73128 | 11 | 128 | 1,2 | 2 | 11.0 | 6.0 |  | 0.25757575757575757 | Patterns of text featuring sequences of tokens that indicate continuation or elaboration, often marked by specific punctuation or formatting. \| A pattern of phrases and terms that suggest actions, roles, or categories, o | advice_or_instruction | low | supported by 2 method(s); advice or instruction |
| 72376 | 11 | 256 | 1 | 1 | 1.0 |  |  | 1.0 | A variety of tokens indicating different parts of speech, including nouns, verbs, and adjectives, often appearing in phrases or clauses that suggest actions, states, or descriptions. | advice_or_instruction | low | supported by 1 method(s); advice or instruction |

## Layer comparison

| layer | unique_candidates | cross_method_overlaps | health_related_candidates | medical_specific_candidates | three_way_consensus_candidates |
| --- | --- | --- | --- | --- | --- |
| 11 | 254 | 22 | 0 | 0 | 4 |
| 15 | 249 | 25 | 0 | 0 | 1 |
| 19 | 219 | 39 | 0 | 0 | 9 |

## Failed and missing interpretation statistics

- Total retained result rows: 900
- Valid interpretations: 613
- Failed interpretations: 287
- Missing interpretations: 0
- Missing all ranking scores: 0
A `not_selected` method status means the latent was absent from that method's top-N; it is not classified as an interpretation failure.

## Recommended shortlist

The shortlist is a transparent prioritization for downstream validation, not a causal claim. See `medical_candidates.csv` for the score, evidence, and full method ranks.

## Limitations

- Interpretations are generated from generic FineWeb activating contexts, not necessarily bad-medical-advice examples.
- The relevance judge score is an EM relevance score and is not itself proof of medical specificity or harmful advice.
- Activating-context sidecars were available for 0 method/latent rows; missing sidecars are not interpreted as evidence.
- Lexical classification is intentionally conservative and should be manually validated before intervention experiments.
- Ranking overlap is limited to supplied CSV rows and configured top-N.
