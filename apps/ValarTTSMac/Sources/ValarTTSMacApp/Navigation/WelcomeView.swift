import SwiftUI

struct WelcomeView: View {
    let onGetStarted: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            hero
            Spacer().frame(height: 48)
            stepCards
            Spacer().frame(height: 44)
            getStartedButton

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 40)
    }

    private var hero: some View {
        VStack(spacing: 20) {
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.10))
                    .frame(width: 88, height: 88)
                Image(systemName: "waveform")
                    .font(.system(size: 38, weight: .medium))
                    .foregroundStyle(Color.accentColor)
            }

            VStack(spacing: 8) {
                Text("Welcome to Valar")
                    .font(.largeTitle.bold())
                Text("Local-first text-to-speech on Apple Silicon.\nNo cloud. No telemetry. Just your voice.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
            }
        }
    }

    private var stepCards: some View {
        HStack(alignment: .top, spacing: 16) {
            OnboardingStepCard(
                step: 1,
                icon: "cpu",
                title: "Choose a model",
                description: "Browse the built-in catalog and install the speech model you want to use."
            )
            OnboardingStepCard(
                step: 2,
                icon: "waveform.badge.plus",
                title: "Generate speech",
                description: "Type or paste text, choose a model, and hit Generate. It's that simple."
            )
            OnboardingStepCard(
                step: 3,
                icon: "person.wave.2",
                title: "Clone your voice",
                description: "Import a short audio sample or describe a voice to build your custom library."
            )
        }
        .frame(maxWidth: 720)
    }

    private var getStartedButton: some View {
        Button(action: onGetStarted) {
            Text("Get Started")
                .frame(minWidth: 140)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .keyboardShortcut(.return, modifiers: [])
    }
}

private struct OnboardingStepCard: View {
    let step: Int
    let icon: String
    let title: String
    let description: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Text("\(step)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white)
                    .frame(width: 22, height: 22)
                    .background(Color.accentColor)
                    .clipShape(Circle())

                Image(systemName: icon)
                    .font(.body.weight(.medium))
                    .foregroundStyle(Color.accentColor)
            }

            Text(title)
                .font(.headline)

            Text(description)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineSpacing(1)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.18))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}
