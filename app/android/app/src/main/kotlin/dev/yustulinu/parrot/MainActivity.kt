package dev.yustulinu.parrot

import android.Manifest
import android.content.pm.PackageManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

// The two things Dart cannot do alone on Android: ask for the microphone, and know the app's folders.
class MainActivity : FlutterActivity() {
    private var pending: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "parrot/platform").setMethodCallHandler { call, result ->
            when (call.method) {
                "folders" -> result.success(
                    mapOf("data" to filesDir.absolutePath, "external" to getExternalFilesDir(null)?.absolutePath),
                )
                "microphone" -> {
                    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                        result.success(true)
                    } else {
                        pending = result
                        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1) {
            pending?.success(grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED)
            pending = null
        }
    }
}
