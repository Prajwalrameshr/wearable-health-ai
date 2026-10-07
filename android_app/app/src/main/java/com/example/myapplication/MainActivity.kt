package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.*
import androidx.health.connect.client.request.AggregateRequest
import androidx.health.connect.client.time.TimeRangeFilter
import androidx.health.connect.client.units.Energy
import androidx.health.connect.client.units.Length
import androidx.work.*
import kotlinx.coroutines.launch
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
            MaterialTheme {
                PermissionScreen(client)
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

@Composable
fun PermissionScreen(client: HealthConnectClient) {
    var text by remember { mutableStateOf("Checking permissions...") }
    var healthData by remember { mutableStateOf<HealthData?>(null) }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

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
            text = "Permission granted!"
            scheduleHealthSync(context)
            scope.launch {
                healthData = readHealthData(client)
            }
        } else {
            text = "Permission denied"
        }
    }

    LaunchedEffect(Unit) {
        val granted = client.permissionController.getGrantedPermissions()
        if (granted.containsAll(permissions)) {
            text = "Permission granted!"
            scheduleHealthSync(context)
            healthData = readHealthData(client)
        } else {
            text = "Permissions not granted. Click button to request."
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Text(text = text, style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))

        if (healthData != null) {
            MetricItem("Total Steps Today", "${healthData?.steps ?: 0}")
            MetricItem("Distance", "${healthData?.distance?.inMeters?.let { "%.2f".format(it / 1000.0) } ?: 0} km")
            MetricItem("Calories Burned", "${healthData?.totalCaloriesBurned?.inKilocalories?.let { "%.0f".format(it) } ?: 0} kcal")
            MetricItem(
                "Heart Rate (Avg Today)",
                "${healthData?.heartRate?.let { "%.0f".format(it) } ?: 0} bpm"
            )
        } else if (text == "Permission granted!") {
            Text(text = "Loading data...")
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                permissionLauncher.launch(permissions)
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Request Permissions / Refresh")
        }
    }
}

@Composable
fun MetricItem(label: String, value: String) {
    Column(modifier = Modifier.padding(vertical = 8.dp)) {
        Text(text = label, style = MaterialTheme.typography.labelLarge)
        Text(text = value, style = MaterialTheme.typography.bodyLarge)
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

fun scheduleHealthSync(context: android.content.Context) {
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
