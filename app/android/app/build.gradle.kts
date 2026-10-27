// The engine (native/) is built by CMake for each ABI; onnxruntime comes prebuilt from its Android
// archive (scripts/fetch-deps.sh unpacks it into native/third_party/onnxruntime-android).
val repo = rootProject.projectDir.resolve("../..")
val abis = listOf("arm64-v8a", "x86_64")

plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "dev.yustulinu.parrot"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = "29.0.14206865"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "dev.yustulinu.parrot"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = 26 // AAudio
        ndk { abiFilters += abis }
        externalNativeBuild {
            cmake {
                // onnxruntime's library uses the shared C++ runtime; so must the engine.
                arguments += listOf("-DANDROID_STL=c++_shared", "-DCMAKE_BUILD_TYPE=Release", "-DPARROT_TESTS=OFF")
                targets += "parrot"
                // Only the 64-bit ABIs (ggml needs fp16 NEON, which 32-bit ARM lacks), whatever Flutter adds.
                abiFilters += abis
            }
        }
        targetSdk = flutter.targetSdkVersion
        // Uses the version code from pubspec.yaml. When using split APKs, 1000 * ABI_VERSION
        // is added automatically by Flutter. (https://developer.android.com/studio/build/configure-apk-splits#configure-APK-versions)
        // You can force using the value of versionCode by specifying the `-P force-version-code-ignoring-abi=true`
        // flag during build.
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    externalNativeBuild {
        cmake {
            path = repo.resolve("native/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    sourceSets["main"].jniLibs.directories += repo.resolve("native/third_party/onnxruntime-android/jni").path
    packaging {
        // From the onnxruntime archive only the 64-bit C library is used (not its Java bindings).
        jniLibs.excludes += listOf("lib/armeabi-v7a/**", "lib/x86/**", "**/libonnxruntime4j_jni.so")
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
