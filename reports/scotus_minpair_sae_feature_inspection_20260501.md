# SCOTUS SAE Feature Example Inspection

## Configuration

| Field | Value |
| --- | --- |
| SAE run | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_sae_probe_20260501_112601 |
| SAE | SAE-Res-Qwen3.5-27B-W80K-L0_100 |
| Region | assistant_all |
| Layer | 8 |
| Min train DF | 2 |
| Source probe dir | sweep_v4/scotus_minpair_replay_20260501_100514 |

## Feature Overview

| Feature | Direction | Weight | Train DF | Active rows | Justice counts | Issue counts |
| --- | --- | --- | --- | --- | --- | --- |
| 49208 | toward_commerce_authority | -4.586 | 5 | 8 | commerce_authority: 8 | Economic Activity: 8 |
| 38148 | toward_commerce_limits | 2.324 | 4 | 4 | commerce_limits: 4 | Economic Activity: 4 |
| 74307 | toward_commerce_authority | -2.305 | 2 | 2 | commerce_authority: 2 | Economic Activity: 2 |
| 13495 | toward_commerce_authority | -2.015 | 9 | 12 | commerce_authority: 8, commerce_limits: 4 | Economic Activity: 12 |
| 16474 | toward_commerce_authority | -2.007 | 5 | 8 | commerce_authority: 8 | Economic Activity: 8 |
| 9967 | toward_commerce_authority | -1.988 | 4 | 7 | commerce_authority: 7 | Economic Activity: 7 |
| 69837 | toward_commerce_authority | -1.982 | 4 | 8 | commerce_authority: 8 | Economic Activity: 8 |
| 28301 | toward_commerce_authority | -1.65 | 3 | 3 | commerce_authority: 3 | Economic Activity: 3 |
| 5939 | toward_commerce_authority | -1.367 | 2 | 4 | commerce_authority: 4 | Economic Activity: 4 |
| 7788 | toward_commerce_authority | -1.286 | 5 | 8 | commerce_authority: 8 | Economic Activity: 8 |
| 6776 | toward_commerce_limits | 1.158 | 5 | 6 | commerce_authority: 3, commerce_limits: 3 | Economic Activity: 6 |
| 35872 | toward_commerce_limits | 1.1 | 5 | 8 | commerce_limits: 8 | Economic Activity: 8 |
| 40707 | toward_commerce_authority | -1.076 | 2 | 3 | commerce_authority: 3 | Economic Activity: 3 |
| 68644 | toward_commerce_authority | -0.8634 | 8 | 9 | commerce_authority: 6, commerce_limits: 3 | Economic Activity: 9 |
| 52914 | toward_commerce_limits | 0.7881 | 9 | 16 | commerce_limits: 16 | Economic Activity: 16 |
| 32007 | toward_commerce_authority | -0.7881 | 3 | 6 | commerce_authority: 4, commerce_limits: 2 | Economic Activity: 6 |
| 54307 | toward_commerce_limits | 0.7603 | 3 | 5 | commerce_limits: 4, commerce_authority: 1 | Economic Activity: 5 |
| 72449 | toward_commerce_limits | 0.6894 | 2 | 4 | commerce_limits: 4 | Economic Activity: 4 |
| 26199 | toward_commerce_authority | -0.6674 | 8 | 16 | commerce_authority: 10, commerce_limits: 6 | Economic Activity: 16 |
| 41577 | toward_commerce_authority | -0.6492 | 17 | 28 | commerce_authority: 17, commerce_limits: 11 | Economic Activity: 28 |

## Top Activating Examples

### Feature 49208 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -4.58631 |
| Train DF | 5 |
| All active rows | 8 |
| Justice counts | commerce_authority: 8 |
| Split counts | train: 5, test: 2, dev: 1 |
| Issue counts | Economic Activity: 8 |
| Posture counts | assistant_replay: 8 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00540026 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00528768 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00473426 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00316773 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00284284 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00241116 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.000912219 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.000489842 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |

### Feature 38148 (toward_commerce_limits)

| Field | Value |
| --- | --- |
| Raw direction label | positive_Ginsburg |
| Weight | 2.32426 |
| Train DF | 4 |
| All active rows | 4 |
| Justice counts | commerce_limits: 4 |
| Split counts | train: 4 |
| Issue counts | Economic Activity: 4 |
| Posture counts | assistant_replay: 4 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0115943 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00804818 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00552704 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00492185 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |

### Feature 74307 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -2.30502 |
| Train DF | 2 |
| All active rows | 2 |
| Justice counts | commerce_authority: 2 |
| Split counts | train: 2 |
| Issue counts | Economic Activity: 2 |
| Posture counts | assistant_replay: 2 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0094287 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00446869 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |

### Feature 13495 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -2.01521 |
| Train DF | 9 |
| All active rows | 12 |
| Justice counts | commerce_authority: 8, commerce_limits: 4 |
| Split counts | train: 9, dev: 2, test: 1 |
| Issue counts | Economic Activity: 12 |
| Posture counts | assistant_replay: 12 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01313 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0105339 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.00780364 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00745426 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.00391989 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.00333141 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00270972 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0026877 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |

### Feature 16474 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -2.00684 |
| Train DF | 5 |
| All active rows | 8 |
| Justice counts | commerce_authority: 8 |
| Split counts | train: 5, test: 2, dev: 1 |
| Issue counts | Economic Activity: 8 |
| Posture counts | assistant_replay: 8 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.019805 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0148568 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0137484 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00941319 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00879554 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00872725 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00585577 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00578531 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |

### Feature 9967 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -1.9884 |
| Train DF | 4 |
| All active rows | 7 |
| Justice counts | commerce_authority: 7 |
| Split counts | train: 4, test: 2, dev: 1 |
| Issue counts | Economic Activity: 7 |
| Posture counts | assistant_replay: 7 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0116604 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0103987 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00888313 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00781906 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00729915 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00361975 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00132072 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |

### Feature 69837 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -1.98179 |
| Train DF | 4 |
| All active rows | 8 |
| Justice counts | commerce_authority: 8 |
| Split counts | train: 4, test: 2, dev: 2 |
| Issue counts | Economic Activity: 8 |
| Posture counts | assistant_replay: 8 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0158167 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0134362 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0116462 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0111324 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.00977636 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0077043 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.00750317 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.00446739 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |

### Feature 28301 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -1.65021 |
| Train DF | 3 |
| All active rows | 3 |
| Justice counts | commerce_authority: 3 |
| Split counts | train: 3 |
| Issue counts | Economic Activity: 3 |
| Posture counts | assistant_replay: 3 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0114227 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.00183012 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 1.06543e-05 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |

### Feature 5939 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -1.36653 |
| Train DF | 2 |
| All active rows | 4 |
| Justice counts | commerce_authority: 4 |
| Split counts | train: 2, dev: 2 |
| Issue counts | Economic Activity: 4 |
| Posture counts | assistant_replay: 4 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0173196 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.00692025 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.00348612 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.000912234 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |

### Feature 7788 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -1.2861 |
| Train DF | 5 |
| All active rows | 8 |
| Justice counts | commerce_authority: 8 |
| Split counts | train: 5, test: 2, dev: 1 |
| Issue counts | Economic Activity: 8 |
| Posture counts | assistant_replay: 8 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0209362 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0197242 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0177463 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0166249 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0156338 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0147379 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0139952 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0083931 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |

### Feature 6776 (toward_commerce_limits)

| Field | Value |
| --- | --- |
| Raw direction label | positive_Ginsburg |
| Weight | 1.15763 |
| Train DF | 5 |
| All active rows | 6 |
| Justice counts | commerce_authority: 3, commerce_limits: 3 |
| Split counts | train: 5, test: 1 |
| Issue counts | Economic Activity: 6 |
| Posture counts | assistant_replay: 6 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0120973 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.010971 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00782672 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.00688273 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00518077 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00046958 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |

### Feature 35872 (toward_commerce_limits)

| Field | Value |
| --- | --- |
| Raw direction label | positive_Ginsburg |
| Weight | 1.10043 |
| Train DF | 5 |
| All active rows | 8 |
| Justice counts | commerce_limits: 8 |
| Split counts | train: 5, test: 2, dev: 1 |
| Issue counts | Economic Activity: 8 |
| Posture counts | assistant_replay: 8 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0329117 | test | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.0292209 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.0142704 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.0138929 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.0105705 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00900787 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.0045438 | test | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |
| 0.00223016 | dev | commerce_limits | Economic Activity | assistant_replay | None | Holding: The statute exceeds Congress's enumerated power. Reasoning: The statute lacks a concrete jurisdictional hook tying the particular conduct to interstate trade. Congress may regulate channels, instrumentalities, and economic activity that substantially affects commerce, but it may not reach every local problem by reciting downstream economic consequences. The federalism structure requires a limiting principle. |

### Feature 40707 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -1.07583 |
| Train DF | 2 |
| All active rows | 3 |
| Justice counts | commerce_authority: 3 |
| Split counts | train: 2, dev: 1 |
| Issue counts | Economic Activity: 3 |
| Posture counts | assistant_replay: 3 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0197747 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0162684 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0151246 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |

### Feature 68644 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -0.863441 |
| Train DF | 8 |
| All active rows | 9 |
| Justice counts | commerce_authority: 6, commerce_limits: 3 |
| Split counts | train: 8, test: 1 |
| Issue counts | Economic Activity: 9 |
| Posture counts | assistant_replay: 9 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0169412 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0121234 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0112812 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0103082 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0100425 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.00670032 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.00588415 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0032699 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |

### Feature 52914 (toward_commerce_limits)

| Field | Value |
| --- | --- |
| Raw direction label | positive_Ginsburg |
| Weight | 0.788088 |
| Train DF | 9 |
| All active rows | 16 |
| Justice counts | commerce_limits: 16 |
| Split counts | train: 9, dev: 4, test: 3 |
| Issue counts | Economic Activity: 16 |
| Posture counts | assistant_replay: 16 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0417858 | dev | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.0361969 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.0350761 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.0338164 | dev | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.0319196 | test | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.0288116 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.0266912 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.0255692 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |

### Feature 32007 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -0.78808 |
| Train DF | 3 |
| All active rows | 6 |
| Justice counts | commerce_authority: 4, commerce_limits: 2 |
| Split counts | train: 3, test: 2, dev: 1 |
| Issue counts | Economic Activity: 6 |
| Posture counts | assistant_replay: 6 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0263007 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0173691 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0166469 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.00484143 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0036107 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |
| 0.000493504 | test | commerce_limits | Economic Activity | assistant_replay | None | Holding: Congress lacks authority on these facts. Reasoning: The regulated conduct is local and non-economic. Lopez and Morrison mark the line between commerce and a general police power. Legislative findings about aggregate economic effects cannot convert a local noneconomic matter into commerce. Accepting that chain of inference would leave no meaningful limit on federal power and would displace traditional state regulation. |

### Feature 54307 (toward_commerce_limits)

| Field | Value |
| --- | --- |
| Raw direction label | positive_Ginsburg |
| Weight | 0.760304 |
| Train DF | 3 |
| All active rows | 5 |
| Justice counts | commerce_limits: 4, commerce_authority: 1 |
| Split counts | train: 3, dev: 1, test: 1 |
| Issue counts | Economic Activity: 5 |
| Posture counts | assistant_replay: 5 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0284835 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0215977 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0192164 | dev | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0165186 | test | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0013469 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |

### Feature 72449 (toward_commerce_limits)

| Field | Value |
| --- | --- |
| Raw direction label | positive_Ginsburg |
| Weight | 0.689447 |
| Train DF | 2 |
| All active rows | 4 |
| Justice counts | commerce_limits: 4 |
| Split counts | train: 2, dev: 1, test: 1 |
| Issue counts | Economic Activity: 4 |
| Posture counts | assistant_replay: 4 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0354432 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0155269 | dev | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0128346 | test | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.00998743 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |

### Feature 26199 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -0.667421 |
| Train DF | 8 |
| All active rows | 16 |
| Justice counts | commerce_authority: 10, commerce_limits: 6 |
| Split counts | train: 8, test: 5, dev: 3 |
| Issue counts | Economic Activity: 16 |
| Posture counts | assistant_replay: 16 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0349244 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0228323 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0194982 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0162172 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0150587 | dev | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0141785 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0124338 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |
| 0.0107331 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The statute is a valid exercise of the commerce power. Reasoning: The relevant activity uses channels or instrumentalities of interstate commerce and forms part of a broader regulatory scheme. Congress may choose civil penalties or statutory damages to make that scheme effective. State remedies may coexist, but traditional state authority does not defeat a valid federal rule directed at interstate commercial networks. |

### Feature 41577 (toward_commerce_authority)

| Field | Value |
| --- | --- |
| Raw direction label | negative_Scalia |
| Weight | -0.649238 |
| Train DF | 17 |
| All active rows | 28 |
| Justice counts | commerce_authority: 17, commerce_limits: 11 |
| Split counts | train: 17, dev: 6, test: 5 |
| Issue counts | Economic Activity: 28 |
| Posture counts | assistant_replay: 28 |

| Activation | Split | Justice | Issue | Posture | Case | Snippet |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0371234 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0365589 | test | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |
| 0.0340402 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0336412 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0333471 | dev | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0326338 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: Congress has authority to regulate this class of activity. Reasoning: The statute targets economic conduct connected to a national market. Wickard and Raich permit Congress to regulate intrastate instances when the class of activity, viewed in the aggregate, substantially affects interstate commerce. The national market would be undermined if local participants could opt out one transaction at a time. |
| 0.0307921 | train | commerce_limits | Economic Activity | assistant_replay | None | Holding: The federal law cannot be sustained under the Commerce Clause. Reasoning: The activity is not part of a broader market regulation of economic production or exchange. It is closer to local crime, family life, education, or property regulation. Those subjects are traditionally governed by the States, and Commerce Clause doctrine does not permit Congress to erase that boundary through attenuated aggregate-effects reasoning. |
| 0.0264707 | train | commerce_authority | Economic Activity | assistant_replay | None | Holding: The federal remedy falls within Congress's Commerce Clause authority. Reasoning: The regulated transactions are commercial in character and tied to interstate systems for distribution, credit, payment, shipping, or market pricing. Congress had a rational basis for treating the local conduct as part of an economic class whose aggregate effects are substantial. |

## Reading Notes

- These are prompt-mean SAE activations from the already-rendered Phase 4 prompts, not fresh token-level SAE traces.
- A feature whose top examples cluster by issue area, procedural posture, named statutes, or boilerplate should be treated as an artifact/confound candidate.
- A feature whose top examples recur across issues and postures while tracking legal reasoning style is a better candidate for deeper inspection.
