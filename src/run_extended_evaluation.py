"""Extended paired robustness evaluation; writes only extended_* outputs."""
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA = Path("data/hc3_original.csv")
PAIRS = Path("results/semantic_paraphrases.csv")
OUT = Path("results")
SEED = 42

def wc(x):
    return len(str(x).split())

def truncate_words(text, n):
    """Truncate a prefix to n whitespace-separated words; do not pad shorter text."""
    text = str(text)
    target = max(1, int(n))
    matches = list(re.finditer(r"\S+", text))
    if not matches or len(matches) <= target:
        return text
    return text[:matches[target - 1].end()].rstrip()

def light_edit(text):
    """Deterministic discourse-marker substitution; not guaranteed meaning-preserving."""
    original = str(text)
    edited = re.sub(r"[ \t]+", " ", original)
    edited = re.sub(r" *\n *", "\n", edited)
    rules = [
        (r"\bIn addition,\s*", "Additionally, "),
        (r"\bAdditionally,\s*", "Also, "),
        (r"\bFor example,\s*", "For instance, "),
        (r"\bFor instance,\s*", "For example, "),
        (r"\bHowever,\s*", "Nevertheless, "),
        (r"\bTherefore,\s*", "Thus, "),
        (r"\bIn conclusion,\s*", "To conclude, "),
        (r"\bOn the other hand,\s*", "Conversely, "),
    ]
    for pattern, replacement in rules:
        updated = re.sub(pattern, replacement, edited, count=1, flags=re.IGNORECASE)
        if updated != edited:
            edited = updated
            break
    return re.sub(r"\s+([,.!?;:])", r"\1", edited).strip() or original

def wilson(k, n, z=1.959963984540054):
    if not n: return [float("nan"), float("nan")]
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h), min(1,c+h)]

def metrics(name, y, pred, changed, ratio):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    n_ai=int((np.asarray(y)==1).sum())
    ci=wilson(int(tp), n_ai)
    return dict(condition=name,n_passages=len(y),n_human=int((np.asarray(y)==0).sum()),
        n_ai=n_ai,accuracy=float(accuracy_score(y,pred)),
        ai_precision=float(precision_score(y,pred,zero_division=0)),
        ai_recall=float(recall_score(y,pred,zero_division=0)),
        ai_f1=float(f1_score(y,pred,zero_division=0)),
        false_positive_rate=float(fp/(fp+tn)) if fp+tn else float("nan"),
        TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
        recall_ci_low=ci[0],recall_ci_high=ci[1],
        ai_passages_changed=int(changed),mean_length_ratio=float(ratio))

def paired_test(name, base, comp, topics):
    a=np.asarray(base)==1; b=np.asarray(comp)==1
    x=int((a & ~b).sum()); y=int((~a & b).sum()); n=x+y
    p=1.0 if n==0 else float(binomtest(min(x,y),n=n,p=.5,alternative="two-sided").pvalue)
    return dict(comparison=name,n_paired_topics=len(topics),detected_to_missed=x,
        missed_to_detected=y,detection_rate_difference_vs_original=float(b.mean()-a.mean()),
        exact_mcnemar_p=p)

def main():
    if not DATA.exists() or not PAIRS.exists():
        raise FileNotFoundError(f"Need both {DATA} and {PAIRS}; run preparation/paraphrase steps first.")
    df=pd.read_csv(DATA)
    req={"topic_id","label","text","source"}
    if req-set(df): raise ValueError(f"Dataset missing {sorted(req-set(df))}")
    df=df.dropna(subset=list(req)).copy()
    df["topic_id"]=df["topic_id"].astype(str); df["label"]=df["label"].astype(int); df["text"]=df["text"].astype(str)
    lab=df.groupby("topic_id")["label"].agg(lambda s:set(s.unique()))
    ids=lab[lab.apply(lambda s:{0,1}.issubset(s))].index
    df=df[df.topic_id.isin(ids)].reset_index(drop=True)
    split=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    tr,te=next(split.split(df,y=df.label,groups=df.topic_id))
    train,test=df.iloc[tr].copy(),df.iloc[te].copy()
    train_ids=set(train.topic_id); test_ids=set(test.topic_id)
    if train_ids & test_ids: raise RuntimeError("Topic leakage detected")
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=200000,
        sublinear_tf=True,strip_accents="unicode"),LinearSVC(class_weight="balanced",random_state=42))
    model.fit(train.text,train.label)
    pairs=pd.read_csv(PAIRS)
    need={"topic_id","source","ai_original","ai_surface_edited","ai_paraphrased","human_text"}
    if need-set(pairs): raise ValueError(f"Paraphrase file missing {sorted(need-set(pairs))}")
    pairs=pairs.dropna(subset=list(need)).copy(); pairs["topic_id"]=pairs.topic_id.astype(str)
    outside=set(pairs.topic_id)-test_ids
    if outside: raise ValueError(f"{len(outside)} paired topics are outside held-out split")
    pairs=pairs.drop_duplicates("topic_id").reset_index(drop=True)
    pairs["ai_light_edited"]=pairs.ai_original.astype(str).map(light_edit)
    pairs["ai_truncated"]=[truncate_words(a,wc(p)) for a,p in zip(pairs.ai_original,pairs.ai_paraphrased)]
    conditions={
      "Original AI":"ai_original",
      "Legacy phrase-removal (existing)":"ai_surface_edited",
      "Verified deterministic light edit":"ai_light_edited",
      "T5 semantic paraphrase":"ai_paraphrased",
      "Truncation control":"ai_truncated"}
    pred_ai={}; rows=[]; pred_frames=[]
    for name,col in conditions.items():
        human=pd.DataFrame({"topic_id":pairs.topic_id,"source":pairs.source,"label":0,
            "text":pairs.human_text.astype(str),"text_type":"Human"})
        ai=pd.DataFrame({"topic_id":pairs.topic_id,"source":pairs.source,"label":1,
            "text":pairs[col].astype(str),"text_type":name})
        ev=pd.concat([human,ai],ignore_index=True)
        ev["prediction"]=model.predict(ev.text); ev["condition"]=name
        ev["correct"]=ev.label.to_numpy()==ev.prediction.to_numpy()
        pred_frames.append(ev)
        pred_ai[name]=ev.loc[ev.label==1,"prediction"].to_numpy()
        ratios=[wc(x)/max(1,wc(y)) for x,y in zip(pairs[col].astype(str),pairs.ai_original.astype(str))]
        changed=(pairs[col].astype(str)!=pairs.ai_original.astype(str)).sum()
        rows.append(metrics(name,ev.label,ev.prediction,changed,np.mean(ratios)))
    md=pd.DataFrame(rows); predictions=pd.concat(pred_frames,ignore_index=True)
    tests=[paired_test(name,pred_ai["Original AI"],pred_ai[name],pairs.topic_id)
        for name in list(conditions)[1:]]
    tests.append(paired_test("T5 paraphrase vs truncation control",
        pred_ai["Truncation control"],pred_ai["T5 semantic paraphrase"],pairs.topic_id))
    td=pd.DataFrame(tests)
    orig=pairs.ai_original.astype(str).map(wc); para=pairs.ai_paraphrased.astype(str).map(wc); trunc=pairs.ai_truncated.astype(str).map(wc)
    ld=pd.DataFrame({"topic_id":pairs.topic_id,"source":pairs.source,"original_words":orig,
        "paraphrase_words":para,"truncation_words":trunc,"paraphrase_to_original_word_ratio":para/orig.replace(0,np.nan)})
    OUT.mkdir(exist_ok=True)
    md.to_csv(OUT/"extended_condition_metrics.csv",index=False)
    predictions.to_csv(OUT/"extended_predictions.csv",index=False)
    td.to_csv(OUT/"extended_paired_tests.csv",index=False)
    ld.to_csv(OUT/"extended_length_analysis.csv",index=False)
    summary={"training_passages":len(train),"training_topics":len(train_ids),"held_out_topics":len(test_ids),
      "paired_questions":len(pairs),"train_test_question_overlap":len(train_ids & test_ids),
      "conditions":conditions,"metrics":md.to_dict("records"),"paired_tests":td.to_dict("records"),
      "length_analysis":{"mean_paraphrase_to_original_word_ratio":float((para/orig.replace(0,np.nan)).mean()),
        "median_paraphrase_to_original_word_ratio":float((para/orig.replace(0,np.nan)).median()),
        "under_60_percent_original_words":int(((para/orig.replace(0,np.nan))<.6).sum()),
        "legacy_phrase_edit_changed_passages":int((pairs.ai_surface_edited.astype(str)!=pairs.ai_original.astype(str)).sum()),
        "deterministic_light_edit_changed_passages":int((pairs.ai_light_edited.astype(str)!=pairs.ai_original.astype(str)).sum()),
        "truncation_control_note":"Original prefix truncated to paraphrase whitespace-word count when original is longer; if original is shorter, it is left unchanged. This is a compression control, not a semantic-preservation control.",
        "truncation_word_count_mismatches":int((trunc != para).sum()),
        "truncation_changed_passages":int((pairs.ai_truncated.astype(str)!=pairs.ai_original.astype(str)).sum()),
        "mean_truncation_to_original_word_ratio":float((trunc/orig.replace(0,np.nan)).mean())}}
    (OUT/"extended_evaluation_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=True))
    try:
        import matplotlib.pyplot as plt
        Path("results/figures").mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(11,5))
        ax.bar(md.condition,md.ai_recall*100); ax.set_ylim(0,100); ax.set_ylabel("AI recall (%)")
        ax.tick_params(axis="x",labelrotation=25,labelsize=8); ax.grid(axis="y",alpha=.25); fig.tight_layout()
        fig.savefig("results/figures/extended_ai_recall.png",dpi=180); plt.close(fig)
    except ImportError: pass
    print("Split: train passages",len(train),"train topics",len(train_ids),"held-out topics",len(test_ids),"overlap",len(train_ids & test_ids))
    print("\nEXTENDED METRICS\n",md[["condition","n_passages","accuracy","ai_recall","false_positive_rate","FN","TP","ai_passages_changed","mean_length_ratio"]].to_string(index=False))
    print("\nPAIRED TESTS\n",td.to_string(index=False))
    print("\nLENGTH SUMMARY\n",json.dumps(summary["length_analysis"],indent=2))
    print("\nSaved extended_* files. Existing final_* outputs were not overwritten.")

if __name__=="__main__":
    main()
