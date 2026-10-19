# Wake word openWakeWord — notes de travail (debug S21 FE)

> Notes de travail maintenues pendant les sessions de debug — mises à jour au fur et à mesure.
> **En cas de compression de contexte, ce fichier est le point de reprise.**
> Dernière mise à jour : 2026-10-18 ~12:10

## Mission

Faire fonctionner le backend openWakeWord du wake word dans l'app Android Hermes Relay
(`com.axiomlabs.hermesrelay.sideload`, branche `wakeword-multi-phrase`) sur le Samsung S21 FE
(`192.168.3.5:5555`, ADB wireless). Symptôme initial : le statut reste **« À vérifier »**
(= `WakeWordRuntimeState.Error`) après activation.

## Architecture du pipeline (vérifiée contre openwakeword officiel)

- 3 modèles TFLite dans `files/wakeword-openwakeword/` :
  `melspectrogram.tflite` (IN `[1,1760]` → OUT `[1,1,8,32]`),
  `embedding_model.tflite` (IN `[1,76,32,1]` → OUT `[1,1,1,96]`),
  `hey_hermes.tflite` (IN `[1,16,96]` ou transposé `[1,96,16]` → score).
- Streaming officiel (`openwakeword/utils.py`) : chunk 1280 samples (80 ms) + 480 de contexte
  → mel 8 frames → buffer roulant **76** frames → 1 embedding par chunk → buffer **120** →
  classifier sur les **16** derniers embeddings (~1,28 s). Le Kotlin réplique exactement
  (`OpenWakeWordDetector.kt`), FGS lit des blocs de 1600 samples, le détecteur rebufférise en
  chunks de 1280 (correct, `remainder`).
- Transform mel : `x/10 + 2` appliqué **dans** le melspectrogram (option `officialTransform`).
- Runtime : `org.tensorflow:tensorflow-lite:2.17.0` (voir §Bug 3 — suspect).

## Bug 1 — chargement des poids Kotlin (FIXÉ)

`MelSpectrogram.kt` lit `mel_weights.bin` (WR 257×512, WI 257×512, MELW 257×32, LE float32).
`asFloatBuffer()` ne déplace pas le curseur du ByteBuffer parent → WI et MELW relisaient le début
du fichier. Fix : sous-vues explicites `ByteBuffer.wrap(weights, offset, count*4)` :
WR @0, WI @`BINS*TAPS*4`, MELW @`BINS*TAPS*8`. Vérifié Java = Python = initializers ONNX.

## Bug 2 — floor de normalisation PAR FRAME (FIXÉ)

openWakeWord normalise la plage dynamique **par frame de 80 ms** : `out[f] = max(F[f], max(F[f]) − 80)`.
Le Kotlin faisait un floor GLOBAL → une frame forte (mel ≈ 1.77e9) écrasait tout (tous les
éléments identiques ≈ 176896256). Fix : boucle par frame dans `MelSpectrogram.transform()`
(`floor = 10f * ln(mv) / LN10 − 80f`, puis `v*0.1f + 2f` si `officialTransform`).
Preuve Python float32 : maxdiff **1.9e-6** vs référence ONNX. Tests `MelSpectrogramTest` : **5/5 PASS**.

## Builds livrés

- APK `hermes-relay-1.18.1-sideload-debug.apk` (221 347 220 octets, 18/10 11:37) installé sur la
  S21 FE (`adb install -r` → Success). Contient les fixes bugs 1+2.
- **MAIS ce build introduit un crash loop (Bug 3 ci-dessous).** Ne pas considérer la tâche finie
  tant que Bug 3 n'est pas résolu.

## Bug 3 — crash loop APP (ACTIF, priorité max) — SIGSEGV natif TFLite 2.17.0

**Symptôme** (confirmé par Rodolphe 18/10 ~12:00) : « depuis le dernier build, l'app ne fait que
crasher ». Crash loop dès que le service wake word tourne ; l'app meurt ~2 s à ~6 min après le
démarrage de la boucle audio.

**Faits (tombstones via `dumpsys dropbox`)** — 18/10 : 11:47:08, 11:47:30, 11:48:15, 11:49:14,
11:55:06 — `data_app_native_crash` :

```
Process: com.axiomlabs.hermesrelay.sideload v59 (1.18.1-sideload)
pid: 23983, tid: DefaultDispatch (worker kotlinx Dispatchers.IO/Default)
signal 11 (SIGSEGV), code 2 (SEGV_ACCERR), fault addr 0x0000007892517040
backtrace (tronqué, code optimisé) :
  #00 pc 0x2f5c20  base.apk!libtensorflowlite_jni.so (offset 0x4990000)
  #01 pc 0x27fa0c  base.apk!libtensorflowlite_jni.so
```

- Registres : marche 4 octets sur des floats (x20–x23 = fault+4..+16) → kernel de compute
  écrivant des floats hors zone protégée.
- `libtensorflowlite_jni.so` arm64-v8a bundlé (build intermediates) :
  **BuildId 850a925e891f910bae3c83095332764b = celui du tombstone** ✓.
  `strings` : **XNNPACK compilé dedans** (125 hits `xnn`/`xnnpack`, APIs
  `TfLiteXNNPackDelegate*`).
- PAS de FATAL EXCEPTION Java, PAS de « Fatal signal » dans logcat — mort silencieuse du
  processus ; ActivityManager « Killing (adj 0): crash ».
- Thread « DefaultDispatch » = la boucle de reconnaissance du `WakeWordForegroundService`
  (scope `Dispatchers.IO`, threads partagés avec Default).

**Chaîne causale** : avant le fix, le melspectrogram TFLite échouait en premier (shape erronée
`[N,1,frames,32]` sur S21 FE) → statut Error « À vérifier » → l'embedding/classifier TFLite
n'avaient **JAMAIS tourné** sur ce téléphone. Maintenant le mel Kotlin fonctionne → l'audio
arrive pour la première fois à l'embedding → **SIGSEGV dans le runtime TFLite**. Le crash arrive
au premier invoke de l'embedding (~0,8 s d'audio ; 10 chunks pour remplir 76 frames) ou peu après
(classifier à ~1,3 s).

**Éliminé** :
- Tailles des buffers : `embWindow` 2432 = `[1,76,32,1]` ✓, `wakeIn` 1536 = `[1,16,96]` ✓,
  `melOut` 8×32 ✓. Pas de lecture hors-tableau côté Java.
- Concurrence : un seul détecteur à la fois (`MicrophoneOwnershipCoordinator`), boucle
  séquentielle dans une coroutine. `Interpreter` non partagé entre threads.
- `resizeInput` dans l'init du détecteur : no-op (le modèle déclare déjà ces shapes), les logs
  on-device confirment « embedding input [1, 76, 32, 1], output [1, 1, 1, 96] » et
  « wake classifier input [1, 16, 96] ».

**Hypothèse principale** : runtime TFLite **2.17.0** bogué sur ce device (Snapdragon 888, arm64).
2.17 a fait de XNNPACK le delegate CPU par défaut pour float32 sur Android → kernel XNNPACK
SIGSEGV sur le modèle d'embedding (conv2d). Cohérent avec le fait que le melspectrogram TFLite
se comportait déjà bizarrement sur ce téléphone.

**Plan de fix (à faire)** :
1. Vérifier la doc/known-issues TFLite 2.17 XNNPACK par défaut (web).
2. Downgrader `gradle/libs.versions.toml` : `tensorflowLite = "2.17.0"` → **2.16.1**
   (dernière avant XNNPACK-défaut ; sémantique `resizeInput` inchangée — déjà vérifiée pour
   2.16 dans T1 ; à re-confirmer).
   OU désactiver XNNPACK explicitement si l'API 2.17 le permet.
3. Rebuild sideloadDebug, réinstaller S21 FE.
4. Vérifier : laisser le FGS écouter ≥ 5 min, zéro tombstone `data_app_native_crash`, le statut
   passe à « Écoute locale » (`wake_word_status_listening`).

**Commandes de vérification crash** :
```
adb -s 192.168.3.5:5555 shell "dumpsys dropbox 2>/dev/null" | tail -25   # liste tombstones
adb -s 192.168.3.5:5555 shell "dumpsys dropbox --print data_app_native_crash" | head -40
adb -s 192.168.3.5:5555 logcat -d | grep -E "has died|Killing.*crash|OpenWakeWordDetector"
```
Scratch : `/home/openclaw/.hermes/cache/scratch/crash_log.txt` (logcat complet 42k lignes),
`tombstone_1155.txt` (tombstone 11:55:06), `relay_settings.preferences_pb` (datastore dump).

## État du téléphone (18/10 ~11:50)

- Modèles **builtin « Hey Hermes »** installés et vérifiés (18/10 09:37) dans
  `files/wakeword-openwakeword/` : mel 1 092 516 B (SHA `96fa0adc…`), embedding 1 330 312 B
  (SHA `c0aea21e…`), `hey_hermes.tflite` 10 519 928 B (spec builtin : GitHub release
  `rodolphebertrand/hermes-relay` tag `wakeword-models-v1`, SHA `6e01e6fb…`), marqueur
  `openwakeword-verified` (« verified\n », 9 B).
- **PAS de `custom_wakeword.tflite` ni `custom_wakeword.meta`** → la tentative « URL + SHA-256 →
  Appliquer » de Rodolphe ce matin n'a **pas persisté** : la clé `wake_word_custom_model_url`
  est ABSENTE du datastore (`relay_settings.preferences_pb`, décodé manuellement). Point à
  clarifier avec Rodolphe (a-t-il cliqué « Effacer » ? l'install custom a-t-il échoué avant la
  sauvegarde ?). Le set builtin marche de toute façon (SHA vérifié).
- Prefs datastore : `wake_word_enabled=true`, phrase « Hey Hermes », sensitivity float 0.2,
  confirmationFrames 2, `assistant_wake_word_enabled=false`.
- Le statut « À vérifier » = `WakeWordRuntimeState.Error` (string `wake_word_status_error`),
  affiché quand `createDetector()` retourne null ou permission micro manquante
  (`HermesVoiceInteractionService.startRecognition` / `WakeWordForegroundService.fail()`).
- Obs. secondaire : au restart 11:56, l'app n'arrive pas à joindre le relay
  `ws://192.168.2.142:9119` (« probeHealth connect refused ») → le gateway Hermes sur :9119
  semble DOWN (à vérifier séparément — indépendant du wake word, mais nécessaire pour l'UX
  post-détection).
- Le serveur de modèles LAN `192.168.2.142:8471` tournait (OK, utilisé ce matin).

## Historique rapide de la session (18/10)

- 09:32 build intermediates regénérés ; 09:37 modèles installés sur le tel (par l'app).
- 11:37 APK 1.18.1 (fixes mel) installé → **crash loop** (tombstones ci-dessus).
- ~11:50 enquête : logcat → « has died » répétés ; dropbox → tombstones natifs ;
  backtrace TFLite ; vérifs buffers/concurrence éliminées ; BuildId .so = tombstone ;
  XNNPACK présent dans le .so.
- ~12:00 Rodolphe confirme le crash loop depuis le dernier build.
- EN COURS : confirmation XNNPACK-défaut en 2.17 (cache Gradle n'a que le POM
  `org.tensorflow:tensorflow-lite:2.17.0` — AAR résolu ailleurs/transformé), puis fix.

## Environnement / rappels build

- Repo : `/home/openclaw/.hermes/projects/hermes-relay-src`, branche `wakeword-multi-phrase`.
- Build : `JAVA_HOME=/home/openclaw/.jdk17 ./gradlew :app:assembleSideloadDebug`
  (APK : `app/build/outputs/apk/sideload/debug/hermes-relay-1.18.1-sideload-debug.apk`).
- Tests : `./gradlew :app:testSideloadDebugUnitTest --tests "com.hermesandroid.relay.wake.MelSpectrogramTest"`.
- Install : `adb -s 192.168.3.5:5555 install -r <apk>`.
- Assets : `app/src/main/assets/mel_weights.bin` (généré depuis initializers ONNX),
  vecteurs `app/src/test/resources/wakeword/mel_test_vectors.json`.
- Réf Python : `/home/openclaw/openwakeword/melspec_ref.py` (onnxruntime), venv `/tmp/onnxvenv`.
- Modèles on-device pulled : `/tmp/devmodels/`.
- PC Windows (`rodol@192.168.2.157`) : `D:\Docker\ow-check` (4 tflite + `shape_test.py`),
  image `tensorflow/tensorflow:2.17.0` pullée — contrôle shapes Docker OPTIONNEL (non bloquant).
  TOUJOURS demander avant de modifier le PC.

## RÉSOLUTION (18/10 ~12:30)

**Bug 3 résolu — cause racine unique de TOUTE la série TFLite** : `Interpreter.run()` recevait
des **FloatArray plats (rank 1)** ; TFLite Java **resize l'input tensor à la shape de l'array
Java reçue** → embedding input devenait [2432] rank-1 → nœud PAD (paddings [4,2]) plantait :
- TFLite **2.17.0** : SIGSEGV natif dans le kernel (crash loop du build 11:37).
- TFLite **2.16.1** : exception Java `pad.cc:123 (4 != 1)` — le message qui a tout révélé.
- L'ancien « mel shape [N,1,frames,32] » = même cause (array plat au mel TFLite).

**Fix** (`OpenWakeWordDetector.kt`) : invokes uniquement via arrays N-D **aux shapes déclarées**
(`float[1][76][32][1]` / `float[1][1][1][96]` / `float[1][16][96]` ou transposé / `float[1][1]`),
buffers plats conservés comme scratch de remplissage. + TFLite 2.16.1 en dép.

**Vérifié on-device 12:21–12:25** : détection « wake phrase detected » ×3 en 3 min, statut
« Écoute… », process stable, 0 tombstone nouveau. FPs = v1 TTS (calibration → v2 URL).
Notes : `/home/openclaw/openwakeword/STATUS.md` § 18/10 (canonique).
Délai écran phone modifié (60 s → 30 min) pour le debug — restaurer après le test live.