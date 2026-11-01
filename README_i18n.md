# MiniMind in other languages

[中文](./README.md) | [English](./README_en.md) | [العربية](#arabic) | [Français](#french) | [Español](#spanish) | [Português](#portuguese) | [Deutsch](#german) | [فارسی](#persian) | [日本語](#japanese) | [한국어](#korean)

These localized introductions provide a concise overview and the fastest path to running MiniMind. The complete and most current technical documentation is maintained in [English](./README_en.md) and [Chinese](./README.md).

<a id="arabic"></a>

<h2 dir="rtl">العربية</h2>

<p dir="rtl">MiniMind هو مشروع مفتوح المصدر يتيح تدريب نموذج لغوي صغير يضم نحو 64 مليون مُعلَمة من الصفر. يمكن إكمال مرحلة الضبط الدقيق الخاضع للإشراف خلال ساعتين تقريبًا على بطاقة NVIDIA 3090 واحدة، بتكلفة استئجار تقارب 3 يوانات صينية.</p>

<ul dir="rtl">
  <li>تنفيذ مبسّط ومباشر باستخدام PyTorch لبنية Dense وMoE.</li>
  <li>مسار تدريب متكامل يشمل التدريب المسبق وSFT وLoRA وDPO وPPO وGRPO وCISPO واستخدام الأدوات وAgentic RL والتقطير.</li>
  <li>دعم Transformers وllama.cpp وvLLM وOllama وواجهة API متوافقة مع OpenAI وواجهة WebUI.</li>
  <li>بيانات وأوزان ونصوص تدريب مفتوحة لتسهيل التعلم وإعادة إنتاج النتائج.</li>
</ul>

<p dir="rtl"><strong>البدء السريع</strong></p>

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

<p dir="rtl">للحصول على تعليمات التدريب ومجموعات البيانات والتقييم وتحويل النموذج، راجع <a href="./README_en.md">الدليل الإنجليزي الكامل</a>.</p>

<a id="french"></a>

## Français

MiniMind est un projet open source permettant d'entraîner de zéro un petit modèle de langage d'environ 64 millions de paramètres. L'étape de fine-tuning supervisé peut être réalisée en environ deux heures sur une seule NVIDIA 3090, pour un coût de location proche de 3 RMB.

- Implémentation PyTorch minimale et lisible des architectures Dense et MoE.
- Pipeline complet : préentraînement, SFT, LoRA, DPO, PPO, GRPO, CISPO, Tool Use, Agentic RL et distillation.
- Compatibilité avec Transformers, llama.cpp, vLLM et Ollama, ainsi qu'une API compatible OpenAI et une WebUI.
- Jeux de données, poids et scripts d'entraînement ouverts pour apprendre et reproduire les résultats.

**Démarrage rapide**

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

Pour les instructions complètes sur l'entraînement, les données, l'évaluation et la conversion des modèles, consultez le [guide complet en anglais](./README_en.md).

<a id="spanish"></a>

## Español

MiniMind es un proyecto de código abierto para entrenar desde cero un modelo de lenguaje pequeño de unos 64 millones de parámetros. La etapa de ajuste supervisado puede completarse en unas dos horas con una sola NVIDIA 3090, con un coste de alquiler cercano a 3 RMB.

- Implementación mínima y legible en PyTorch de arquitecturas Dense y MoE.
- Flujo completo: preentrenamiento, SFT, LoRA, DPO, PPO, GRPO, CISPO, uso de herramientas, Agentic RL y destilación.
- Compatibilidad con Transformers, llama.cpp, vLLM y Ollama, además de una API compatible con OpenAI y una WebUI.
- Conjuntos de datos, pesos y scripts de entrenamiento abiertos para aprender y reproducir resultados.

**Inicio rápido**

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

Para obtener instrucciones completas sobre entrenamiento, datos, evaluación y conversión de modelos, consulta la [documentación completa en inglés](./README_en.md).

<a id="portuguese"></a>

## Português

MiniMind é um projeto de código aberto para treinar do zero um pequeno modelo de linguagem com cerca de 64 milhões de parâmetros. A etapa de ajuste fino supervisionado pode ser concluída em aproximadamente duas horas com uma única NVIDIA 3090, por um custo de locação próximo de 3 RMB.

- Implementação PyTorch mínima e legível de arquiteturas Dense e MoE.
- Pipeline completo: pré-treinamento, SFT, LoRA, DPO, PPO, GRPO, CISPO, uso de ferramentas, Agentic RL e destilação.
- Compatibilidade com Transformers, llama.cpp, vLLM e Ollama, além de API compatível com OpenAI e WebUI.
- Conjuntos de dados, pesos e scripts de treinamento abertos para aprendizado e reprodução dos resultados.

**Início rápido**

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

Para instruções completas sobre treinamento, dados, avaliação e conversão de modelos, consulte a [documentação completa em inglês](./README_en.md).

<a id="german"></a>

## Deutsch

MiniMind ist ein Open-Source-Projekt, mit dem sich ein kleines Sprachmodell mit rund 64 Millionen Parametern von Grund auf trainieren lässt. Das überwachte Fine-Tuning kann auf einer einzelnen NVIDIA 3090 in ungefähr zwei Stunden abgeschlossen werden; die entsprechenden Mietkosten liegen bei etwa 3 RMB.

- Kleine, gut lesbare PyTorch-Implementierung von Dense- und MoE-Architekturen.
- Vollständige Pipeline: Pretraining, SFT, LoRA, DPO, PPO, GRPO, CISPO, Tool Use, Agentic RL und Distillation.
- Kompatibel mit Transformers, llama.cpp, vLLM und Ollama; zusätzlich stehen eine OpenAI-kompatible API und eine WebUI bereit.
- Offene Datensätze, Gewichte und Trainingsskripte zum Lernen und Reproduzieren der Ergebnisse.

**Schnellstart**

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

Vollständige Anleitungen zu Training, Datensätzen, Evaluation und Modellkonvertierung enthält die [englische Dokumentation](./README_en.md).

<a id="persian"></a>

<h2 dir="rtl">فارسی</h2>

<p dir="rtl">MiniMind یک پروژهٔ متن‌باز برای آموزش یک مدل زبانی کوچک با حدود ۶۴ میلیون پارامتر از صفر است. مرحلهٔ تنظیم دقیق نظارت‌شده را می‌توان با یک کارت NVIDIA 3090 در حدود دو ساعت و با هزینهٔ اجاره‌ای نزدیک به ۳ یوان چین انجام داد.</p>

<ul dir="rtl">
  <li>پیاده‌سازی ساده و خوانای معماری‌های Dense و MoE با PyTorch.</li>
  <li>زنجیرهٔ کامل آموزش شامل پیش‌آموزش، SFT، LoRA، DPO، PPO، GRPO، CISPO، استفاده از ابزار، Agentic RL و تقطیر مدل.</li>
  <li>سازگار با Transformers، llama.cpp، vLLM و Ollama، همراه با API سازگار با OpenAI و WebUI.</li>
  <li>داده‌ها، وزن‌ها و اسکریپت‌های آموزشی باز برای یادگیری و بازتولید نتایج.</li>
</ul>

<p dir="rtl"><strong>شروع سریع</strong></p>

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

<p dir="rtl">برای دستورالعمل کامل آموزش، داده‌ها، ارزیابی و تبدیل مدل، <a href="./README_en.md">مستندات کامل انگلیسی</a> را ببینید.</p>

<a id="japanese"></a>

## 日本語

MiniMind は、約 6,400 万パラメータの小規模言語モデルをゼロから学習できるオープンソースプロジェクトです。教師ありファインチューニングは、NVIDIA 3090 1枚で約2時間、レンタル費用約3人民元で完了できます。

- Dense および MoE アーキテクチャを、読みやすい最小限の PyTorch コードで実装。
- 事前学習、SFT、LoRA、DPO、PPO、GRPO、CISPO、Tool Use、Agentic RL、蒸留までを網羅。
- Transformers、llama.cpp、vLLM、Ollama に対応し、OpenAI互換APIとWebUIも提供。
- 学習と再現のためのデータセット、重み、学習スクリプトを公開。

**クイックスタート**

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

学習、データセット、評価、モデル変換の詳しい手順については、[英語版の完全なドキュメント](./README_en.md)を参照してください。

<a id="korean"></a>

## 한국어

MiniMind는 약 6,400만 개의 파라미터를 가진 소형 언어 모델을 처음부터 학습할 수 있는 오픈 소스 프로젝트입니다. 지도 미세 조정 단계는 NVIDIA 3090 한 장에서 약 2시간, 약 3위안의 대여 비용으로 완료할 수 있습니다.

- Dense 및 MoE 아키텍처를 간결하고 읽기 쉬운 PyTorch 코드로 구현합니다.
- 사전 학습, SFT, LoRA, DPO, PPO, GRPO, CISPO, 도구 사용, Agentic RL, 모델 증류를 포함한 전체 파이프라인을 제공합니다.
- Transformers, llama.cpp, vLLM, Ollama와 호환되며 OpenAI 호환 API와 WebUI를 제공합니다.
- 학습과 결과 재현을 위한 데이터셋, 가중치, 학습 스크립트를 공개합니다.

**빠른 시작**

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
python eval_llm.py --load_from ./minimind-3
```

학습, 데이터셋, 평가, 모델 변환에 대한 전체 안내는 [영문 문서](./README_en.md)를 참고하세요.

---

These translations are community-maintained summaries. If you find an inaccurate phrase or want to maintain a complete translation, please open a pull request.
