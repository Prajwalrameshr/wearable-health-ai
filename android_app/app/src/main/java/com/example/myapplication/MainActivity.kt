package com.example.myapplication

import android.content.Context
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.*
import androidx.health.connect.client.request.AggregateRequest
import androidx.health.connect.client.time.TimeRangeFilter
import androidx.health.connect.client.units.Energy
import androidx.health.connect.client.units.Length
import androidx.work.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.Instant
import java.time.ZonedDateTime
import java.time.temporal.ChronoUnit
import java.util.concurrent.TimeUnit

class MainActivity : ComponentActivity() {

    private lateinit var client: HealthConnectClient

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        client = HealthConnectClient.getOrCreate(this)

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    background = Color(0xFF070B14),
                    surface = Color(0xFF0F172A),
                    primary = Color(0xFF06B6D4),
                    secondary = Color(0xFF10B981)
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    WearableHealthAppScreen(client)
                }
            }
        }
    }
}

data class HealthData(
    val steps: Long?,
    val distance: Length?,
    val totalCaloriesBurned: Energy?,
    val heartRate: Double?
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WearableHealthAppScreen(client: HealthConnectClient) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var statusText by remember { mutableStateOf("Checking permissions...") }
    var healthData by remember { mutableStateOf<HealthData?>(null) }
    var serverUrl by remember { mutableStateOf(ApiClient.getBaseUrl(context)) }
    var isSyncing by remember { mutableStateOf(false) }
    var syncResult by remember { mutableStateOf<HealthResponse?>(null) }
    var syncError by remember { mutableStateOf<String?>(null) }

    val permissions = setOf(
        HealthPermission.getReadPermission(StepsRecord::class),
        HealthPermission.getReadPermission(DistanceRecord::class),
        HealthPermission.getReadPermission(TotalCaloriesBurnedRecord::class),
        HealthPermission.getReadPermission(HeartRateRecord::class)
    )

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = PermissionController.createRequestPermissionResultContract()
    ) { granted ->
        if (granted.containsAll(permissions)) {
            statusText = "Health Connect Permissions Granted!"
            scheduleHealthSync(context)
            scope.launch {
                healthData = readHealthData(client)
            }
        } else {
            statusText = "Permissions partially granted or denied."
        }
    }

    LaunchedEffect(Unit) {
        val granted = client.permissionController.getGrantedPermissions()
        if (granted.containsAll(permissions)) {
            statusText = "Health Connect Connected"
            scheduleHealthSync(context)
            healthData = readHealthData(client)
        } else {
            statusText = "Tap below to grant Health Connect sensor permissions"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // App Header Banner
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "♥ ",
                        color = Color(0xFFEF4444),
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Wearable Health AI",
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.ExtraBold
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Smartwatch Health Connect Telemetry Ingestion",
                    color = Color(0xFF94A3B8),
                    fontSize = 12.sp
                )
            }
        }

        // Server Backend Configuration
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF131D33)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "🔗 AI Backend Server URL",
                    color = Color(0xFF38BDF8),
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = serverUrl,
                    onValueChange = { serverUrl = it },
                    label = { Text("FastAPI Server URL (Port 5000)") },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFF06B6D4),
                        unfocusedBorderColor = Color(0xFF334155),
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White
                    ),
                    singleLine = true
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = {
                            ApiClient.setBaseUrl(context, serverUrl)
                            Toast.makeText(context, "Server URL Saved!", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0284C7))
                    ) {
                        Text("Save URL", fontSize = 12.sp)
                    }
                    OutlinedButton(
                        onClick = {
                            serverUrl = "http://10.0.2.2:5000/"
                            ApiClient.setBaseUrl(context, serverUrl)
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Emulator", fontSize = 12.sp)
                    }
                }
            }
        }

        // Status Card
        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = "● $statusText",
                color = if (statusText.contains("Connected") || statusText.contains("Granted")) Color(0xFF10B981) else Color(0xFFF59E0B),
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(12.dp)
            )
        }

        // Biometric Metrics Card
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "📊 Today's Smartwatch Sensor Data",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                )
                Spacer(modifier = Modifier.height(12.dp))

                val stepsVal = healthData?.steps ?: 0L
                val distKm = healthData?.distance?.inMeters?.let { "%.2f".format(it / 1000.0) } ?: "0.00"
                val calVal = healthData?.totalCaloriesBurned?.inKilocalories?.let { "%.0f".format(it) } ?: "0"
                val hrVal = healthData?.heartRate?.let { "%.0f".format(it) } ?: "--"

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    MetricBadge("🚶 Steps", "$stepsVal")
                    MetricBadge("📍 Distance", "$distKm km")
                }
                Spacer(modifier = Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    MetricBadge("🔥 Calories", "$calVal kcal")
                    MetricBadge("♥ Avg Heart Rate", "$hrVal bpm")
                }
            }
        }

        // Action Buttons
        Button(
            onClick = {
                permissionLauncher.launch(permissions)
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF334155))
        ) {
            Text("Request / Refresh Watch Permissions")
        }

        Button(
            onClick = {
                isSyncing = true
                syncError = null
                syncResult = null
                scope.launch {
                    try {
                        val startOfDay = ZonedDateTime.now().truncatedTo(ChronoUnit.DAYS).toInstant()
                        val now = Instant.now()
                        val deviceUserId = Settings.Secure.getString(
                            context.contentResolver,
                            Settings.Secure.ANDROID_ID
                        ) ?: "android_user"

                        val distM = healthData?.distance?.inMeters
                        val distKm = if (distM != null) distM / 1000.0 else 2.5
                        val cal = healthData?.totalCaloriesBurned?.inKilocalories ?: 1850.0
                        val hr = healthData?.heartRate ?: 68.0

                        val payload = HealthPayload(
                            deviceUserId = deviceUserId,
                            steps = healthData?.steps ?: 4500L,
                            distanceKm = distKm,
                            distanceMeters = distM,
                            calories = cal,
                            caloriesKcal = cal,
                            heartRate = hr,
                            averageHeartRate = hr,
                            recordStartTime = startOfDay.toString(),
                            recordEndTime = now.toString(),
                            collectedAt = now.toString(),
                            city = "Bangalore",
                            modelType = "gmm"
                        )

                        val api = ApiClient.getApi(context)
                        val resp = withContext(Dispatchers.IO) {
                            api.sendHealthData(payload)
                        }

                        if (resp.isSuccessful && resp.body() != null) {
                            syncResult = resp.body()
                            Toast.makeText(context, "AI Analysis Synchronized!", Toast.LENGTH_SHORT).show()
                        } else {
                            syncError = "Server Error ${resp.code()}: ${resp.message()}"
                        }
                    } catch (e: Exception) {
                        syncError = "Connection Failed: ${e.localizedMessage}"
                    } finally {
                        isSyncing = false
                    }
                }
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF06B6D4)),
            enabled = !isSyncing
        ) {
            Text(
                text = if (isSyncing) "⚡ Analyzing with AI Engine..." else "⚡ Sync to AI Backend Now",
                fontWeight = FontWeight.Bold,
                fontSize = 15.sp
            )
        }

        // Live AI Prediction Response Card
        if (syncResult != null) {
            val res = syncResult!!
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0D1E3A)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "🧠 Live AI Engine Prediction",
                        color = Color(0xFF38BDF8),
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                    Spacer(modifier = Modifier.height(10.dp))

                    val stateColor = when (res.state) {
                        "Recovery" -> Color(0xFF10B981)
                        "Baseline" -> Color(0xFFF59E0B)
                        else -> Color(0xFFEF4444)
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(text = "Predicted State:", color = Color(0xFF94A3B8))
                        Text(
                            text = res.state ?: "Unknown",
                            color = stateColor,
                            fontWeight = FontWeight.Bold,
                            fontSize = 18.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(6.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(text = "Strain Index Score:", color = Color(0xFF94A3B8))
                        Text(
                            text = "${"%.1f".format(res.riskScore ?: 0.0)} / 100",
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Spacer(modifier = Modifier.height(6.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(text = "Clinical Advisory:", color = Color(0xFF94A3B8))
                        Text(
                            text = res.clinicalAdvisoryLevel ?: "Normal",
                            color = if (res.clinicalAdvisoryLevel == "Normal") Color(0xFF10B981) else Color(0xFFF59E0B),
                            fontWeight = FontWeight.Bold
                        )
                    }

                    if (!res.recommendations.isNullOrEmpty()) {
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "💡 Directive: ${res.recommendations!!.first()}",
                            color = Color(0xFFE2E8F0),
                            fontSize = 13.sp
                        )
                    }
                }
            }
        }

        if (syncError != null) {
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF3B151E)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "⚠️ $syncError\n(Ensure FastAPI is running: python backend/main.py)",
                    color = Color(0xFFF87171),
                    fontSize = 13.sp,
                    modifier = Modifier.padding(12.dp)
                )
            }
        }
    }
}

@Composable
fun MetricBadge(label: String, value: String) {
    Card(
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
        modifier = Modifier.width(160.dp)
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Text(text = label, color = Color(0xFF94A3B8), fontSize = 12.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
        }
    }
}

suspend fun readHealthData(client: HealthConnectClient): HealthData? {
    return try {
        val startOfDay = ZonedDateTime.now().truncatedTo(ChronoUnit.DAYS).toInstant()
        val now = Instant.now()

        val response = client.aggregate(
            AggregateRequest(
                metrics = setOf(
                    StepsRecord.COUNT_TOTAL,
                    DistanceRecord.DISTANCE_TOTAL,
                    TotalCaloriesBurnedRecord.ENERGY_TOTAL,
                    HeartRateRecord.BPM_AVG
                ),
                timeRangeFilter = TimeRangeFilter.between(startOfDay, now)
            )
        )

        HealthData(
            steps = response[StepsRecord.COUNT_TOTAL],
            distance = response[DistanceRecord.DISTANCE_TOTAL],
            totalCaloriesBurned = response[TotalCaloriesBurnedRecord.ENERGY_TOTAL],
            heartRate = response[HeartRateRecord.BPM_AVG]?.toDouble()
        )
    } catch (e: Exception) {
        null
    }
}

fun scheduleHealthSync(context: Context) {
    val constraints = Constraints.Builder()
        .setRequiredNetworkType(NetworkType.CONNECTED)
        .build()

    val syncRequest = PeriodicWorkRequestBuilder<HealthWorker>(15, TimeUnit.MINUTES)
        .setConstraints(constraints)
        .build()

    WorkManager.getInstance(context).enqueueUniquePeriodicWork(
        "HealthSyncWorker",
        ExistingPeriodicWorkPolicy.UPDATE,
        syncRequest
    )
}
