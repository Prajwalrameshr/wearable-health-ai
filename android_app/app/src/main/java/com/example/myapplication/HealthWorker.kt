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

// Define your API interface
interface HealthApi {
    @POST("api/health/records")
    suspend fun sendHealthData(@Body data: HealthPayload): retrofit2.Response<Unit>
}

// Define the data payload
data class HealthPayload(
    val deviceUserId: String,
    val steps: Long,
    val distanceMeters: Double?,
    val caloriesKcal: Double?,
    val averageHeartRate: Double?,
    val recordStartTime: String,
    val recordEndTime: String,
    val collectedAt: String
)

class HealthWorker(appContext: Context, workerParams: WorkerParameters) :
    CoroutineWorker(appContext, workerParams) {

    private val healthConnectClient by lazy { HealthConnectClient.getOrCreate(appContext) }

    // Setup Retrofit (Replace BASE_URL with your actual backend URL)
    private val retrofit = Retrofit.Builder()
        .baseUrl("http://192.168.116.121:8080/")
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    private val api = retrofit.create(HealthApi::class.java)

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
            ) ?: "unknown-device"

            val payload = HealthPayload(
                deviceUserId = deviceUserId,
                steps = response[StepsRecord.COUNT_TOTAL] ?: 0L,
                distanceMeters = response[DistanceRecord.DISTANCE_TOTAL]?.inMeters ?: 0.0,
                caloriesKcal = response[TotalCaloriesBurnedRecord.ENERGY_TOTAL]?.inKilocalories ?: 0.0,
                averageHeartRate = response[HeartRateRecord.BPM_AVG]?.toDouble() ?: 0.0,
                recordStartTime = startOfDay.toString(),
                recordEndTime = now.toString(),
                collectedAt = now.toString()
            )

            // Send to backend
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
