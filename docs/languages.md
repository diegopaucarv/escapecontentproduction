# Catálogo de idiomas — spaCy y Stanza

Generado por `scripts/fetch_language_catalog.py`. spaCy compatible: **3.8**.

## spaCy — idiomas con pipeline entrenado

| Código | Modelos |
| --- | --- |
| **ca** | ca_core_news_lg (v['3.8.0']), ca_core_news_md (v['3.8.0']), ca_core_news_sm (v['3.8.0']), ca_core_news_trf (v['3.8.0']) |
| **da** | da_core_news_lg (v['3.8.0']), da_core_news_md (v['3.8.0']), da_core_news_sm (v['3.8.0']), da_core_news_trf (v['3.8.0']) |
| **de** | de_core_news_lg (v['3.8.0']), de_core_news_md (v['3.8.0']), de_core_news_sm (v['3.8.0']), de_dep_news_trf (v['3.8.0']) |
| **el** | el_core_news_lg (v['3.8.0']), el_core_news_md (v['3.8.0']), el_core_news_sm (v['3.8.0']) |
| **en** | en_core_web_lg (v['3.8.0']), en_core_web_md (v['3.8.0']), en_core_web_sm (v['3.8.0']), en_core_web_trf (v['3.8.0']), en_core_web_hftrf (v['3.8.0']) |
| **es** | es_core_news_lg (v['3.8.0']), es_core_news_md (v['3.8.0']), es_core_news_sm (v['3.8.0']), es_dep_news_trf (v['3.8.0']) |
| **fi** | fi_core_news_lg (v['3.8.0']), fi_core_news_md (v['3.8.0']), fi_core_news_sm (v['3.8.0']) |
| **fr** | fr_core_news_lg (v['3.8.0']), fr_core_news_md (v['3.8.0']), fr_core_news_sm (v['3.8.0']), fr_dep_news_trf (v['3.8.0']) |
| **hr** | hr_core_news_lg (v['3.8.0']), hr_core_news_md (v['3.8.0']), hr_core_news_sm (v['3.8.0']) |
| **it** | it_core_news_lg (v['3.8.0']), it_core_news_md (v['3.8.0']), it_core_news_sm (v['3.8.0']) |
| **ja** | ja_core_news_lg (v['3.8.0']), ja_core_news_md (v['3.8.0']), ja_core_news_sm (v['3.8.0']), ja_core_news_trf (v['3.8.0']) |
| **ko** | ko_core_news_lg (v['3.8.0']), ko_core_news_md (v['3.8.0']), ko_core_news_sm (v['3.8.0']) |
| **lt** | lt_core_news_lg (v['3.8.0']), lt_core_news_md (v['3.8.0']), lt_core_news_sm (v['3.8.0']) |
| **mk** | mk_core_news_lg (v['3.8.0']), mk_core_news_md (v['3.8.0']), mk_core_news_sm (v['3.8.0']) |
| **nb** | nb_core_news_lg (v['3.8.0']), nb_core_news_md (v['3.8.0']), nb_core_news_sm (v['3.8.0']) |
| **nl** | nl_core_news_lg (v['3.8.0']), nl_core_news_md (v['3.8.0']), nl_core_news_sm (v['3.8.0']) |
| **pl** | pl_core_news_lg (v['3.8.0']), pl_core_news_md (v['3.8.0']), pl_core_news_sm (v['3.8.0']) |
| **pt** | pt_core_news_lg (v['3.8.0']), pt_core_news_md (v['3.8.0']), pt_core_news_sm (v['3.8.0']) |
| **ro** | ro_core_news_lg (v['3.8.0']), ro_core_news_md (v['3.8.0']), ro_core_news_sm (v['3.8.0']) |
| **ru** | ru_core_news_lg (v['3.8.0']), ru_core_news_md (v['3.8.0']), ru_core_news_sm (v['3.8.0']) |
| **sl** | sl_core_news_lg (v['3.8.0']), sl_core_news_md (v['3.8.0']), sl_core_news_sm (v['3.8.0']), sl_core_news_trf (v['3.8.0']) |
| **sv** | sv_core_news_lg (v['3.8.0']), sv_core_news_md (v['3.8.0']), sv_core_news_sm (v['3.8.0']) |
| **uk** | uk_core_news_lg (v['3.8.0']), uk_core_news_md (v['3.8.0']), uk_core_news_sm (v['3.8.0']), uk_core_news_trf (v['3.8.0']) |
| **xx** | xx_ent_wiki_sm (v['3.8.0']), xx_sent_ud_sm (v['3.8.0']) |
| **zh** | zh_core_web_lg (v['3.8.0']), zh_core_web_md (v['3.8.0']), zh_core_web_sm (v['3.8.0']), zh_core_web_trf (v['3.8.0']) |

## Stanza — idiomas con modelos UD

| Código | Procesadores | Coref |
| --- | --- | --- |
| **ab** | depparse, lemma, pos, tokenize | — |
| **af** | depparse, lemma, ner, pos, tokenize | — |
| **ang** | depparse, lemma, ner, pos, tokenize | — |
| **ar** | depparse, lemma, mwt, ner, pos, tokenize | — |
| **be** | depparse, lemma, pos, tokenize | — |
| **bg** | depparse, lemma, ner, pos, tokenize | — |
| **bn** |  | — |
| **bxr** | depparse, lemma, pos, tokenize | — |
| **ca** | coref, depparse, lemma, mwt, pos, tokenize | ✅ |
| **cop** | depparse, lemma, mwt, pos, tokenize | — |
| **cs** | coref, depparse, lemma, mwt, pos, tokenize | ✅ |
| **cu** | depparse, lemma, pos, tokenize | — |
| **cy** | depparse, lemma, mwt, pos, tokenize | — |
| **da** | constituency, depparse, lemma, ner, pos, tokenize | — |
| **de** | constituency, coref, depparse, lemma, mwt, ner, pos, sentiment, tokenize | ✅ |
| **el** | depparse, lemma, mwt, pos, tokenize | — |
| **en** | constituency, coref, depparse, lemma, mwt, ner, pos, sentiment, tokenize | ✅ |
| **es** | constituency, coref, depparse, lemma, mwt, ner, pos, sentiment, tokenize | ✅ |
| **et** | depparse, lemma, mwt, pos, tokenize | — |
| **eu** | depparse, lemma, pos, tokenize | — |
| **fa** | depparse, lemma, mwt, ner, pos, tokenize | — |
| **fi** | depparse, lemma, mwt, ner, pos, tokenize | — |
| **fo** | depparse, mwt, pos, tokenize | — |
| **fr** | coref, depparse, lemma, mwt, ner, pos, tokenize | ✅ |
| **fro** | depparse, lemma, pos, tokenize | — |
| **ga** | depparse, lemma, pos, tokenize | — |
| **gd** | depparse, lemma, mwt, pos, tokenize | — |
| **gl** | depparse, lemma, mwt, pos, tokenize | — |
| **got** | depparse, lemma, pos, tokenize | — |
| **grc** | depparse, lemma, mwt, pos, tokenize | — |
| **gv** | depparse, lemma, mwt, pos, tokenize | — |
| **hbo** | depparse, lemma, mwt, pos, tokenize | — |
| **he** | coref, depparse, lemma, mwt, ner, pos, tokenize | ✅ |
| **hi** | coref, depparse, lemma, ner, pos, tokenize | ✅ |
| **hr** | depparse, lemma, pos, tokenize | — |
| **hsb** | depparse, lemma, pos, tokenize | — |
| **hu** | depparse, lemma, ner, pos, tokenize | — |
| **hy** | depparse, lemma, mwt, ner, pos, tokenize | — |
| **hyw** | depparse, lemma, mwt, pos, tokenize | — |
| **id** | constituency, depparse, lemma, mwt, pos, tokenize | — |
| **is** | depparse, lemma, mwt, pos, tokenize | — |
| **it** | constituency, depparse, lemma, mwt, ner, pos, tokenize | — |
| **ja** | constituency, depparse, lemma, ner, pos, tokenize | — |
| **ka** | depparse, lemma, mwt, pos, tokenize | — |
| **kk** | depparse, lemma, mwt, ner, pos, tokenize | — |
| **kmr** | depparse, lemma, mwt, pos, tokenize | — |
| **ko** | depparse, lemma, pos, tokenize | — |
| **kpv** | depparse, lemma, mwt, pos, tokenize | — |
| **ky** | depparse, lemma, pos, tokenize | — |
| **la** | depparse, lemma, mwt, pos, tokenize | — |
| **lij** | depparse, lemma, mwt, pos, tokenize | — |
| **lt** | depparse, lemma, pos, tokenize | — |
| **lv** | depparse, lemma, pos, tokenize | — |
| **lzh** | depparse, lemma, pos, tokenize | — |
| **ml** |  | — |
| **mr** | depparse, lemma, mwt, ner, pos, sentiment, tokenize | — |
| **mt** | depparse, pos, tokenize | — |
| **multilingual** | langid | — |
| **my** | ner, tokenize | — |
| **myv** | depparse, lemma, mwt, pos, tokenize | — |
| **nb** | coref, depparse, lemma, ner, pos, tokenize | ✅ |
| **nds** | depparse, lemma, pos, tokenize | — |
| **nl** | depparse, lemma, mwt, ner, pos, tokenize | — |
| **nn** | coref, depparse, lemma, ner, pos, tokenize | ✅ |
| **or** | ner | — |
| **orv** | depparse, lemma, pos, tokenize | — |
| **ota** | depparse, lemma, mwt, pos, tokenize | — |
| **pcm** | depparse, lemma, pos, tokenize | — |
| **pl** | coref, depparse, lemma, mwt, ner, pos, tokenize | ✅ |
| **pt** | constituency, depparse, lemma, mwt, pos, tokenize | — |
| **qaf** | depparse, lemma, mwt, pos, tokenize | — |
| **qpm** | depparse, lemma, pos, tokenize | — |
| **qtd** | depparse, lemma, mwt, pos, tokenize | — |
| **ro** | depparse, lemma, pos, tokenize | — |
| **ru** | coref, depparse, lemma, ner, pos, tokenize | ✅ |
| **sa** | depparse, lemma, pos, tokenize | — |
| **sd** | depparse, lemma, ner, pos, tokenize | — |
| **si** |  | — |
| **sk** | depparse, lemma, mwt, pos, tokenize | — |
| **sl** | depparse, lemma, pos, tokenize | — |
| **sme** | depparse, lemma, pos, tokenize | — |
| **sq** | depparse, lemma, mwt, pos, tokenize | — |
| **sr** | depparse, lemma, pos, tokenize | — |
| **sv** | depparse, lemma, ner, pos, tokenize | — |
| **ta** | coref, depparse, lemma, mwt, pos, tokenize | ✅ |
| **te** | depparse, ner, pos, tokenize | — |
| **th** | depparse, ner, pos, tokenize | — |
| **tr** | constituency, depparse, lemma, mwt, ner, pos, tokenize | — |
| **ug** | depparse, lemma, pos, tokenize | — |
| **uk** | depparse, lemma, mwt, ner, pos, tokenize | — |
| **ur** | depparse, lemma, ner, pos, tokenize | — |
| **vi** | constituency, depparse, ner, pos, sentiment, tokenize | — |
| **wo** | depparse, lemma, mwt, pos, tokenize | — |
| **xcl** | depparse, lemma, mwt, pos, tokenize | — |
| **zh-hans** | constituency, depparse, lemma, ner, pos, sentiment, tokenize | — |
| **zh-hant** | depparse, lemma, pos, tokenize | — |

## Notas

- El segmentador pide a Stanza los procesadores `tokenize,pos,lemma,depparse,constituency,coref`.
- El coref de Stanza **solo** existe para: ca, cs, de, en, es, fr, he, hi, nb, nn, pl, ru, ta.
- Para idiomas sin coref, el segmentador usa spaCy y desactiva la correferencia (try/except en `get_stanza`).
