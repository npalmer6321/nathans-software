plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.selfforge.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.selfforge.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}
