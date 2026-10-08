package com.example.myapplication

import android.content.Context
import android.provider.Settings
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.records.*
import androidx.health.connect.client.request.AggregateRequest
import androidx.health.connect.client.time.TimeRangeFilter
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST
import java.time.Instant
import java.time.ZonedDateTime
import java.time.temporal.ChronoUnit

// Define your API interface aligned with FastAPI Backend
interface HealthApi {
    @POST("api/health/records")
    suspend fun sendHealthData(@Body data: HealthPayload): retrofit2.Response<HealthResponse>
}

// Data payload matching backend/main.py HealthPayload schema
data class HealthPayload(
    val deviceUserId: String,
    val steps: Long = 0,
    val distanceKm: Double? = null,
    val distanceMeters: Double? = null,
    val calories: Double? = null,
    val caloriesKcal: Double? = null,
    val heartRate: Double? = null,
    val averageHeartRate: Double? = null,
    val heartRateResting: Double? = null,
    val hrvRmssdAvg: Double? = null,
    val oxygenSaturation: Double? = null,
    val oxygenSaturationNadir: Double? = null,
    val sleepMinutes: Int = 0,
    val recordStartTime: String? = null,
    val recordEndTime: String? = null,
    val collectedAt: String? = null,
    val city: String? = "Bangalore",
    val modelType: String? = "gmm"
)

// Response schema matching backend/main.py HealthResponse
data class HealthResponse(
    val status: String,
    val message: String? = null,
    val days_available: Int = 0,
    val window_used: String? = null,
    val userId: String? = null,
    val date: String? = null,
    val modelType: String? = null,
    val state: String? = null,
    val previousState: String? = null,
    val confidence: Double? = null,
    val trend: String? = null,
    val riskScore: Double? = null,
    val riskLevel: String? = null,
    val severityScore: Double? = null,
    val clinicalAdvisoryLevel: String? = null,
    val clinicalSummaryMessage: String? = null,
    val recommendations: List<String>? = null
)

// Dynamic Retrofit API Client with SharedPreferences backing
object ApiClient {
    private const val PREFS_NAME = "HealthAppPrefs"
    private const val KEY_BASE_URL = "backend_base_url"
    const val DEFAULT_BASE_URL = "http://10.0.2.2:5000/"

    fun getBaseUrl(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        var url = prefs.getString(KEY_BASE_URL, DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL
        if (!url.endsWith("/")) {
            url += "/"
        }
        return url
    }

    fun setBaseUrl(context: Context, url: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        var formatted = url.trim()
        if (!formatted.endsWith("/")) {
            formatted += "/"
        }
        prefs.edit().putString(KEY_BASE_URL, formatted).apply()
    }

    fun getApi(context: Context): HealthApi {
        val baseUrl = getBaseUrl(context)
        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(HealthApi::class.java)
    }
}

class HealthWorker(appContext: Context, workerParams: WorkerParameters) :
    CoroutineWorker(appContext, workerParams) {

    private val healthConnectClient by lazy { HealthConnectClient.getOrCreate(appContext) }

    override suspend fun doWork(): Result {
        return try {
            val now = Instant.now()
            val startOfDay = ZonedDateTime.now().truncatedTo(ChronoUnit.DAYS).toInstant()

            val response = healthConnectClient.aggregate(
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

            val deviceUserId = Settings.Secure.getString(
                applicationContext.contentResolver,
                Settings.Secure.ANDROID_ID
            ) ?: "android_device_user"

            val stepsVal = response[StepsRecord.COUNT_TOTAL] ?: 0L
            val distMeters = response[DistanceRecord.DISTANCE_TOTAL]?.inMeters
            val distKm = if (distMeters != null) distMeters / 1000.0 else null
            val calsVal = response[TotalCaloriesBurnedRecord.ENERGY_TOTAL]?.inKilocalories
            val hrVal = response[HeartRateRecord.BPM_AVG]?.toDouble()

            val payload = HealthPayload(
                deviceUserId = deviceUserId,
                steps = stepsVal,
                distanceKm = distKm,
                distanceMeters = distMeters,
                calories = calsVal,
                caloriesKcal = calsVal,
                heartRate = hrVal,
                averageHeartRate = hrVal,
                recordStartTime = startOfDay.toString(),
                recordEndTime = now.toString(),
                collectedAt = now.toString(),
                modelType = "gmm"
            )

            val api = ApiClient.getApi(applicationContext)
            val apiResponse = api.sendHealthData(payload)

            if (apiResponse.isSuccessful) {
                Result.success()
            } else {
                Result.retry()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Result.failure()
        }
    }
}
