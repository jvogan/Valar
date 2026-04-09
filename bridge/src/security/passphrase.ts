/**
 * Passphrase extraction hook for channel voice security.
 *
 * Config lives in the bridge layer; core runtime is unaware.
 * Disabled by default - pass `undefined` as config to skip all checks.
 */

export interface PassphraseConfig {
  phrase: string;
  requireAlignerVerification?: boolean;
  confidenceThreshold?: number;
  maxGapSeconds?: number;
}

export interface AlignmentToken {
  text: string;
  startTime: number;
  endTime: number;
  confidence?: number | null;
}

export interface AlignmentResult {
  transcript: string;
  tokens: AlignmentToken[];
}

export interface PassphraseExtractionResult {
  passphraseFound: boolean;
  confidence: number | null;
  strippedTranscript: string;
}

function stripPhrase(transcript: string, phrase: string): string {
  const idx = transcript.toLowerCase().indexOf(phrase.toLowerCase());
  if (idx === -1) return transcript;
  return (transcript.slice(0, idx) + transcript.slice(idx + phrase.length))
    .replace(/\s{2,}/g, " ")
    .trim();
}

function normalise(word: string): string {
  return word.toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/gi, "");
}

function findPhraseTokenRun(
  tokens: AlignmentToken[],
  phraseWords: string[],
): AlignmentToken[] | null {
  if (phraseWords.length === 0) return null;
  for (let i = 0; i <= tokens.length - phraseWords.length; i++) {
    const run = tokens.slice(i, i + phraseWords.length);
    const allMatch = run.every(
      (tok, j) => normalise(tok.text) === normalise(phraseWords[j]!),
    );
    if (allMatch) return run;
  }
  return null;
}

/**
 * Extract and verify a passphrase from a transcript, optionally using
 * forced-alignment data for timing and confidence verification.
 *
 * Usage in bridge pipelines:
 * ```ts
 * const result = extractPassphrase(transcript, channelConfig.passphrase, alignment);
 * if (!result.passphraseFound) { rejectMessage(); return; }
 * forwardTranscript(result.strippedTranscript);
 * ```
 */
export function extractPassphrase(
  transcript: string,
  config: PassphraseConfig | undefined,
  alignment?: AlignmentResult,
): PassphraseExtractionResult {
  if (!config) {
    return {
      passphraseFound: false,
      confidence: null,
      strippedTranscript: transcript,
    };
  }

  const {
    phrase,
    requireAlignerVerification = false,
    confidenceThreshold = 0.6,
    maxGapSeconds = 0.5,
  } = config;

  const phraseInTranscript = transcript
    .toLowerCase()
    .includes(phrase.toLowerCase());

  if (!phraseInTranscript) {
    return {
      passphraseFound: false,
      confidence: null,
      strippedTranscript: transcript,
    };
  }

  if (requireAlignerVerification) {
    if (!alignment) {
      return {
        passphraseFound: false,
        confidence: null,
        strippedTranscript: transcript,
      };
    }

    const phraseWords = phrase.trim().split(/\s+/).filter(Boolean);
    const run = findPhraseTokenRun(alignment.tokens, phraseWords);

    if (!run) {
      return {
        passphraseFound: false,
        confidence: null,
        strippedTranscript: transcript,
      };
    }

    for (let i = 1; i < run.length; i++) {
      const gap = run[i]!.startTime - run[i - 1]!.endTime;
      if (gap > maxGapSeconds) {
        return {
          passphraseFound: false,
          confidence: null,
          strippedTranscript: transcript,
        };
      }
    }

    const reportedConfidences = run
      .map((tok) => tok.confidence)
      .filter((c): c is number => c !== undefined && c !== null);

    if (reportedConfidences.length > 0) {
      const minConfidence = Math.min(...reportedConfidences);
      if (minConfidence < confidenceThreshold) {
        return {
          passphraseFound: false,
          confidence: minConfidence,
          strippedTranscript: transcript,
        };
      }
    }

    const avgConfidence =
      reportedConfidences.length > 0
        ? reportedConfidences.reduce((a, b) => a + b, 0) /
          reportedConfidences.length
        : null;

    return {
      passphraseFound: true,
      confidence: avgConfidence,
      strippedTranscript: stripPhrase(transcript, phrase),
    };
  }

  return {
    passphraseFound: true,
    confidence: null,
    strippedTranscript: stripPhrase(transcript, phrase),
  };
}
