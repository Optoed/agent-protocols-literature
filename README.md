# Автоматический выбор протокола взаимодействия LLM-агентов

Литобзор и каталог 22 публикаций: выбор готовых протоколов, построение агентных систем и управление взаимодействием во время исполнения. Подборка сформирована к 28 сентября 2026 года; обзор обновлён 5 ноября 2026 года. Включены 20 конференционных работ вне Findings и две публикации Findings.

**Автор обзора: Petr Stekliannikov.**

[Полный литературный обзор](LITERATURE_REVIEW.md) · [Индекс статей](index.tsv)

Названия ведут к PDF; рядом даны текстовая версия, публикация и авторский репозиторий, если ссылка известна. Классификация предложена в обзоре, классы могут пересекаться. Результаты описаны по статьям; сторонние реализации не запускались.

## A. Выбор готового протокола

1. **[MDAgents](pdf/mdagents.pdf)** — NeurIPS 2024 main. Оценивает сложность медицинского запроса, выбирает режим решения и подбирает специалистов; внутри группового режима организует обсуждение. [Текст](md/mdagents.md) · [Публикация](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html) · [Код](https://github.com/mitmedialab/MDAgents)

2. **[MasRouter](pdf/masrouter.pdf)** — ACL 2025 main. Обучает каскадный маршрутизатор, который выбирает режим кооперации, роли и модели с учётом качества и стоимости. [Текст](md/masrouter.md) · [Публикация](https://aclanthology.org/2025.acl-long.757/) · [Код](https://github.com/yanweiyue/MasRouter)

3. **[BiCSRouter](pdf/bicsrouter.pdf)** — Findings ACL 2026. Выбирает между глубоким рассуждением одного агента и сотрудничеством нескольких. Использует подбор конфигураций внутри режимов и прогнозы качества и стоимости; часть сравнительных результатов заимствована из MasRouter. [Текст](md/bicsrouter.md) · [Публикация](https://aclanthology.org/2026.findings-acl.947/)

## B. Генерация и оптимизация структуры

4. **[GPTSwarm](pdf/gptswarm.pdf)** — ICML 2024 main. Представляет агентов вычислительным графом и отдельно оптимизирует промпты узлов и вероятности межагентных связей. Статья допускает обновление графа во время работы. [Текст](md/gptswarm.md) · [Публикация](https://proceedings.mlr.press/v235/zhuge24a.html) · [Код](https://github.com/metauto-ai/GPTSwarm)

5. **[ADAS](pdf/adas.pdf)** — ICLR 2025 main. Meta Agent Search генерирует исполнимый код агентных систем, оценивает кандидатов и использует архив результатов при создании новых вариантов. [Текст](md/adas.md) · [Публикация](https://proceedings.iclr.cc/paper_files/paper/2025/file/36b7acf6f6010652b3f2a433774a66fe-Paper-Conference.pdf) · [Код](https://github.com/ShengranHu/ADAS)

6. **[AFlow](pdf/aflow.pdf)** — ICLR 2025 main. Использует MCTS-подобный выбор кандидатов, изменения схем с помощью LLM и оценивание на валидационной выборке. Поиск ведётся в пространстве кода и готовых операторов. [Текст](md/aflow.md) · [Публикация](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5492ecbce4439401798dcd2c90be94cd-Abstract-Conference.html) · [Код](https://github.com/FoundationAgents/AFlow)

7. **[AgentSquare](pdf/agentsquare.pdf)** — ICLR 2025 main. Ищет модульную архитектуру агента через рекомбинацию и эволюцию компонентов. Прогноз качества получает от LLM по примерам оценённых систем в контексте; область поиска шире межагентной коммуникации. [Текст](md/agentsquare.md) · [Публикация](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0ae94013da7cd459402fd77874e09ee3-Abstract-Conference.html) · [Код](https://github.com/tsinghua-fib-lab/AgentSquare)

8. **[Cognify](pdf/cognify.pdf)** — KDD 2025 research. Оптимизирует структуру и параметры GenAI-схем с помощью AdaSeek: иерархического TPE и адаптивного распределения внутреннего бюджета. [Текст](md/cognify.md) · [Публикация](https://doi.org/10.1145/3711896.3736884) · [Код](https://github.com/GenseeAI/cognify)

9. **[AutoPDL](pdf/autopdl.pdf)** — AutoML 2025. Применяет последовательное отсеивание (successive halving) к декларативным конфигурациям промптов: шаблонам, инструкциям и демонстрациям. Не выполняет свободную генерацию межагентных графов. [Текст](md/autopdl.md) · [Публикация](https://proceedings.mlr.press/v293/spiess25a.html) · [Код](https://github.com/IBM/prompt-declaration-language)

10. **[MaAS](pdf/maas.pdf)** — ICML 2025. Обучает распределение операторов агентной суперсети и выбирает подграф и глубину под запрос. Выбор зависит от предыдущих выбранных операторов, что не равнозначно использованию их фактических ответов. [Текст](md/maas.md) · [Публикация](https://proceedings.mlr.press/v267/zhang25bi.html) · [Код](https://github.com/bingreeky/MaAS)

11. **[G-Designer](pdf/gdesigner.pdf)** — ICML 2025 main. Строит коммуникационный граф из представлений задачи и агентов с помощью вариационного графового кодировщика и декодера, затем прореживает связи. [Текст](md/gdesigner.md) · [Публикация](https://proceedings.mlr.press/v267/zhang25cu.html) · [Код](https://github.com/yanweiyue/GDesigner)

12. **[MAS-GPT](pdf/masgpt.pdf)** — ICML 2025 main. Дообучает генератор на парах «запрос–агентная система», чтобы получать исполнимый MAS-код за один вызов. Эта стоимость не включает подготовку данных, обучение и исполнение созданной системы. [Текст](md/masgpt.md) · [Публикация](https://proceedings.mlr.press/v267/ye25g.html) · [Код](https://github.com/rui-ye/MAS-GPT)

13. **[AgentPrune](pdf/agentprune.pdf)** — ICLR 2025. Обучает пространственные и временные маски рёбер, затем однократно прореживает коммуникационный граф. Сокращает существующие связи, а не выбирает следующего агента по текущим ответам. [Текст](md/agentprune.md) · [Публикация](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbc461518c59a2a8d64e70e2c38c4a0e-Abstract-Conference.html) · [Код](https://github.com/yanweiyue/AgentPrune)

14. **[SwarmAgentic](pdf/swarmagentic.pdf)** — EMNLP 2025 main. Совместно оптимизирует функции агентов и их взаимодействие через PSO-подобные обновления популяции с учётом индивидуальных и общих лучших результатов. [Текст](md/swarmagentic.md) · [Публикация](https://aclanthology.org/2025.emnlp-main.93/) · [Код](https://github.com/YaoZ720/SwarmAgenticCode)

15. **[AgentSwift](pdf/agentswift.pdf)** — AAAI 2026 technical track. Использует иерархический поиск с отдельной обучаемой моделью предсказания качества, направляющей выбор перспективных агентных систем. [Текст](md/agentswift.md) · [Публикация](https://ojs.aaai.org/index.php/AAAI/article/view/40453) · [Код](https://github.com/Ericccc02/AgentSwift)

16. **[A²Flow](pdf/a2flow.pdf)** — AAAI 2026 technical track. Извлекает операторы из примеров, кластеризует, обобщает и сохраняет их для построения схем. Меняет пространство операторов, а не только правило поиска. [Текст](md/a2flow.md) · [Публикация](https://ojs.aaai.org/index.php/AAAI/article/view/40240) · [Код](https://github.com/pandawei-ele/A2FLOW)

17. **[BayesFlow](pdf/bayesflow.pdf)** — Findings EACL 2026. Вероятностно строит частичные схемы, оценивая их продолжения; полная система включает дополнительный модуль улучшения кандидатов. Теоретический результат относится к процедуре без этого модуля. [Текст](md/bayesflow.md) · [Публикация](https://aclanthology.org/2026.findings-eacl.165/) · [Код](https://github.com/BoYuanVisionary/BayesFlow)

18. **[Grammar Search](pdf/grammar.pdf)** — ACL 2026 main. Строит допустимые композиции компонентов по контекстно-свободной грамматике без LLM-генерации кода, принудительно включая редко выбранные компоненты. В настройке авторов MCTS и Thompson sampling не дали выигрыша. [Текст](md/grammar.md) · [Публикация](https://aclanthology.org/2026.acl-long.75/) · [Код](https://github.com/mayanks43/grammar_search)

19. **[CARD](pdf/card.pdf)** — ICLR 2026 main. Генерирует коммуникационный граф с учётом свойств агентов и условий среды; заявляет повторную перестройку при изменении условий. Эксперименты в разных средах не равнозначны проверке восстановления после каждого сбоя внутри задачи. [Текст](md/card.md) · [Публикация](https://iclr.cc/virtual/2026/papers.html) · [Код](https://github.com/Warma10032/CARD)

## C. Изменение взаимодействия во время исполнения

20. **[DyLAN](pdf/dylan.pdf)** — COLM 2024 main. Ранжирует промежуточные ответы, отбирает активных агентов следующего раунда и останавливает обсуждение по согласованности. Также включает предварительную оптимизацию команды. [Текст](md/dylan.md) · [Публикация](https://openreview.net/pdf?id=XII0Wp1XA9) · [Код](https://github.com/SALT-NLP/DyLAN)

21. **[Puppeteer](pdf/puppeteer.pdf)** — NeurIPS 2025 main. Обучает централизованную политику выбирать следующего агента и завершение по текущему состоянию задачи, учитывая качество и стоимость траектории. [Текст](md/puppeteer.md) · [Публикация](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f1320d2e2842169c6fc89dcbd80e94d0-Abstract-Conference.html) · [Код](https://github.com/OpenBMB/ChatDev/tree/puppeteer)

22. **[SupervisorAgent](pdf/supervisor.pdf)** — ICLR 2026 main. Фильтр риска без LLM-вызовов определяет, когда нужен контроллер: разрешить продолжение, дать указание, исправить наблюдение или вызвать проверку. Экономия токенов в статье сопровождается ростом задержки. [Текст](md/supervisor.md) · [Публикация](https://openreview.net/forum?id=pzFhtpkabh) · [Код](https://github.com/LINs-lab/SupervisorAgent)

## Материалы

- `pdf/` — сохранённые PDF статей.
- `md/` — извлечённые тексты для поиска и чтения; формулы и таблицы следует сверять с PDF. Некоторые тексты получены из авторских arXiv-версий; источник указан в начале файла.
- `index.tsv` — названия, конференции, классификация и ссылки.

Формат каталога основан на [aaai26-demos](https://github.com/dmitryglhf/aaai26-demos). Статьи принадлежат их авторам и правообладателям; ссылки на оригиналы приведены для каждой работы.
