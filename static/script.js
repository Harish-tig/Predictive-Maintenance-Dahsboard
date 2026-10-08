// Global variables to store charts
let rulProgressionChart = null;
let healthDistributionChart = null;
let rulHistogramChart = null;
let urgencyChart = null;

// Global variable to store all classification results
let allClassificationResults = {};

// ==================== FILE INPUT HANDLERS ====================

// Misalignment file inputs
document
  .getElementById("misalignTempFile")
  .addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "No file chosen";
    document.getElementById("misalignTempFileName").textContent = fileName;
  });

document
  .getElementById("misalignVibFile")
  .addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "No file chosen";
    document.getElementById("misalignVibFileName").textContent = fileName;
  });

// BPFI file inputs
document
  .getElementById("bpfiTempFile")
  .addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "No file chosen";
    document.getElementById("bpfiTempFileName").textContent = fileName;
  });

document.getElementById("bpfiVibFile").addEventListener("change", function (e) {
  const fileName = e.target.files[0]?.name || "No file chosen";
  document.getElementById("bpfiVibFileName").textContent = fileName;
});

// ==================== FORM SUBMISSION HANDLERS ====================

// Misalignment form submission
document
  .getElementById("misalignForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    await handleFormSubmission("misalign");
  });

// BPFI form submission
document
  .getElementById("bpfiForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    await handleFormSubmission("bpfi");
  });

// ==================== RUL PREDICTION HANDLERS ====================

// Generic form submission handler for RUL prediction
async function handleFormSubmission(modelType) {
  const tempFileId = `${modelType}TempFile`;
  const vibFileId = `${modelType}VibFile`;
  const btnId = `${modelType}PredictBtn`;
  const btnTextId = `${modelType}BtnText`;
  const btnLoaderId = `${modelType}BtnLoader`;

  const tempFile = document.getElementById(tempFileId).files[0];
  const vibFile = document.getElementById(vibFileId).files[0];

  if (!tempFile || !vibFile) {
    showAlert("error", "Please select both files");
    return;
  }

  // Show loading state
  const predictBtn = document.getElementById(btnId);
  const btnText = document.getElementById(btnTextId);
  const btnLoader = document.getElementById(btnLoaderId);

  predictBtn.disabled = true;
  btnText.style.display = "none";
  btnLoader.style.display = "inline-block";

  // Prepare form data
  const formData = new FormData();
  formData.append("temp_file", tempFile);
  formData.append("vib_file", vibFile);
  formData.append("model_type", modelType);

  try {
    // Send request to Flask backend
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      showAlert(
        "success",
        `Predictions generated successfully using ${modelType} model!`,
      );
      displayResults(data);
    } else {
      showAlert("error", data.message || data.error || "Prediction failed");
    }
  } catch (error) {
    console.error("Error:", error);
    showAlert("error", "An error occurred during prediction");
  } finally {
    // Reset button state
    predictBtn.disabled = false;
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
  }
}

// ==================== DISPLAY RUL RESULTS ====================

function displayResults(data) {
  console.log("Displaying results:", data);

  const { predictions, statistics, classifications } = data;

  // Validate data
  if (!predictions || !statistics) {
    showAlert("error", "Invalid response data");
    return;
  }

  // Show model info
  const modelInfo = document.getElementById("modelInfo");
  const modelBadge = document.getElementById("modelBadge");
  modelInfo.style.display = "block";
  modelBadge.textContent = `RUL Model: ${(
    statistics.model_type || "unknown"
  ).toUpperCase()}`;
  modelBadge.className = `model-badge model-badge-${
    statistics.model_type || "misalign"
  }`;

  // Display classification results if available
  console.log("Classifications object:", classifications);
  if (classifications && Object.keys(classifications).length > 0) {
    displayClassificationCards(classifications);
  } else {
    console.warn("No classification results to display");
    // Hide classification section if no results
    document.getElementById("classificationSection").style.display = "none";
  }

  // Show sections
  document.getElementById("statsSection").style.display = "grid";
  document.getElementById("chartsSection").style.display = "grid";

  // Update statistics cards with safe defaults
  document.getElementById("totalSequences").textContent =
    statistics.total_sequences || 0;
  document.getElementById("meanRUL").textContent = `${(
    statistics.mean_rul_minutes || 0
  ).toFixed(1)} min`;
  document.getElementById("warningCount").textContent =
    statistics.warning_count || 0;
  document.getElementById("severeCount").textContent =
    statistics.severe_count || 0;
  document.getElementById("criticalCount").textContent =
    statistics.critical_count || 0;

  // Update detailed statistics table with safe defaults
  document.getElementById("detailMeanRUL").textContent = `${(
    statistics.mean_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("detailMedianRUL").textContent = `${(
    statistics.median_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("minRUL").textContent = `${(
    statistics.min_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("maxRUL").textContent = `${(
    statistics.max_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("rangeRUL").textContent = `${(
    statistics.range_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("stdRUL").textContent = `${(
    statistics.std_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("varianceRUL").textContent = `${(
    statistics.variance_rul_minutes || 0
  ).toFixed(2)}`;
  document.getElementById("q1RUL").textContent = `${(
    statistics.q1_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("q3RUL").textContent = `${(
    statistics.q3_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("iqrRUL").textContent = `${(
    statistics.iqr_rul_minutes || 0
  ).toFixed(2)} min`;
  document.getElementById("healthyCount").textContent =
    statistics.healthy_count || 0;

  // [MODIFIED] RUL Progression chart animates point-by-point at 0.3s delay.
  // All other charts render immediately as before.
  createHealthDistributionChart(statistics);
  createRULHistogramChart(predictions.rul_minutes || []);
  createUrgencyChart(statistics);
  animateRULProgressionChart(predictions); // ← replaces createRULProgressionChart

  // Show alerts based on conditions
  const criticalCount = statistics.critical_count || 0;
  const severeCount = statistics.severe_count || 0;

  if (criticalCount > 0) {
    showAlert(
      "warning",
      `🚨 CRITICAL: ${criticalCount} sequences need immediate maintenance!`,
    );
  } else if (severeCount > 0) {
    showAlert(
      "warning",
      `⚠️ WARNING: ${severeCount} sequences in severe condition. Plan maintenance soon.`,
    );
  }
}

// ==================== [NEW] ANIMATED RUL PROGRESSION CHART ====================
// Receives the full predictions object, builds the line chart incrementally.
// Each point is added every 300 ms — the delay is purely visual; all data
// arrived in the single /predict response.

function animateRULProgressionChart(predictions) {
  const data = predictions.rul_minutes || [];
  const indices = predictions.sequence_indices || [];

  if (data.length === 0) return;

  // Destroy existing chart
  if (rulProgressionChart) {
    rulProgressionChart.destroy();
    rulProgressionChart = null;
  }

  // Size canvas to final width up-front so the scroll container works
  const canvas = document.getElementById("rulProgressionChart");
  const minWidth = Math.max(800, indices.length * 15);
  canvas.style.width = `${minWidth}px`;

  const ctx = canvas.getContext("2d");

  // Create chart with empty data arrays
  rulProgressionChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Predicted RUL (minutes)",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 3,
      // Disable Chart.js built-in animation so our interval controls timing
      animation: { duration: 0 },
      plugins: {
        legend: { display: true, position: "top" },
        tooltip: {
          callbacks: {
            label: function (context) {
              const rul = context.parsed.y.toFixed(2);
              const hours = (context.parsed.y / 60).toFixed(2);
              return `RUL: ${rul} min (${hours} hrs)`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "RUL (minutes)",
            font: { size: 14, weight: "bold" },
          },
          ticks: { callback: (v) => v.toFixed(0) },
        },
        x: {
          title: {
            display: true,
            text: "Sequence Index",
            font: { size: 14, weight: "bold" },
          },
          ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 20 },
        },
      },
    },
  });

  // Feed one point every 300 ms
  let i = 0;
  const DELAY_MS = 300;

  const timer = setInterval(() => {
    if (i >= data.length) {
      clearInterval(timer);
      return;
    }
    rulProgressionChart.data.labels.push(indices[i]);
    rulProgressionChart.data.datasets[0].data.push(data[i]);
    rulProgressionChart.update("none"); // "none" = skip animation per tick
    i++;
  }, DELAY_MS);
}

// ==================== DISPLAY CLASSIFICATION CARDS ====================

function displayClassificationCards(classifications) {
  const classificationSection = document.getElementById(
    "classificationSection",
  );
  const cardsContainer = document.getElementById("classificationCards");

  if (!classifications || Object.keys(classifications).length === 0) {
    console.log("No classifications to display");
    classificationSection.style.display = "none";
    return;
  }

  console.log(
    "Displaying classification cards for:",
    Object.keys(classifications),
  );
  classificationSection.style.display = "block";
  cardsContainer.innerHTML = "";

  // Check if we have ensemble results
  if (classifications.ensemble) {
    console.log("Found ensemble classification:", classifications.ensemble);
    displayEnsembleCard(classifications.ensemble, cardsContainer);
  } else {
    console.warn("No ensemble key found in classifications");
  }
}

function displayEnsembleCard(result, container) {
  console.log("Creating ensemble card with result:", result);

  // Determine severity class for styling
  let severityClass = "normal";
  let severityIcon = "✅";

  if (result.fault_type !== "Normal") {
    if (result.maintenance_recommendation.includes("CRITICAL")) {
      severityClass = "critical";
      severityIcon = "🚨";
    } else if (result.maintenance_recommendation.includes("WARNING")) {
      severityClass = "warning";
      severityIcon = "⚠️";
    } else {
      severityClass = "caution";
      severityIcon = "⚡";
    }
  }

  // Create card HTML
  const card = document.createElement("div");
  card.className = `classification-card ${severityClass}`;

  // Build class distribution HTML if available
  let distributionHTML = "";
  if (
    result.class_distribution &&
    Object.keys(result.class_distribution).length > 0
  ) {
    distributionHTML =
      '<div class="class-distribution"><h4>📊 Prediction Distribution:</h4><ul>';
    for (const [className, data] of Object.entries(result.class_distribution)) {
      distributionHTML += `<li><strong>${className}:</strong> ${data.count} segments (${data.percentage.toFixed(1)}%)</li>`;
    }
    distributionHTML += "</ul></div>";
  }

  card.innerHTML = `
    <div class="classification-header">
      <h3>🤖 ${result.model_type || "Ensemble Model"}</h3>
      <span class="confidence-badge">
        ${(result.confidence * 100).toFixed(1)}% Confidence
      </span>
    </div>

    <div class="classification-body">
      <div class="classification-item">
        <span class="item-label">Predicted Class</span>
        <span class="item-value class-name">${result.predicted_class}</span>
      </div>

      <div class="classification-item">
        <span class="item-label">Fault Type</span>
        <span class="item-value fault-type">${result.fault_description}</span>
      </div>

      <div class="classification-item">
        <span class="item-label">Severity</span>
        <span class="item-value severity-badge severity-${severityClass}">
          ${severityIcon} ${result.severity}
        </span>
      </div>

      <div class="classification-item full-width">
        <span class="item-label">Maintenance Action</span>
        <div class="maintenance-box">
          <p>${result.maintenance_recommendation}</p>
        </div>
      </div>

      ${distributionHTML}

      <div class="classification-stats">
        <div class="stat-small">
          <span class="stat-label">Total Segments Analyzed</span>
          <span class="stat-value">${result.total_segments_analyzed}</span>
        </div>
      </div>
    </div>
  `;

  container.appendChild(card);
  console.log("Ensemble card added to container");
}

// ==================== CREATE CHARTS ====================

function createHealthDistributionChart(statistics) {
  const ctx = document
    .getElementById("healthDistributionChart")
    .getContext("2d");

  if (healthDistributionChart) {
    healthDistributionChart.destroy();
  }

  const healthyCount = statistics.healthy_count || 0;
  const warningCount = statistics.warning_count || 0;
  const severeCount = statistics.severe_count || 0;
  const criticalCount = statistics.critical_count || 0;

  healthDistributionChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Healthy", "Warning", "Severe", "Critical"],
      datasets: [
        {
          data: [healthyCount, warningCount, severeCount, criticalCount],
          backgroundColor: ["#22c55e", "#f59e0b", "#ff6b35", "#ef4444"],
          borderWidth: 2,
          borderColor: "#ffffff",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            padding: 15,
            font: {
              size: 12,
            },
            generateLabels: function (chart) {
              const data = chart.data;
              if (data.labels.length && data.datasets.length) {
                return data.labels.map((label, i) => {
                  const value = data.datasets[0].data[i];
                  const total = data.datasets[0].data.reduce(
                    (a, b) => a + b,
                    0,
                  );
                  const percentage =
                    total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                  return {
                    text: `${label}: ${value} (${percentage}%)`,
                    fillStyle: data.datasets[0].backgroundColor[i],
                    hidden: false,
                    index: i,
                  };
                });
              }
              return [];
            },
          },
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const total = statistics.total_sequences || 1;
              const value = context.parsed;
              const percentage = ((value / total) * 100).toFixed(1);
              return `${context.label}: ${value} (${percentage}%)`;
            },
          },
        },
      },
    },
  });
}

function createRULHistogramChart(rulData) {
  const ctx = document.getElementById("rulHistogramChart").getContext("2d");

  if (rulHistogramChart) {
    rulHistogramChart.destroy();
  }

  if (!rulData || rulData.length === 0) {
    console.warn("No data available for RUL histogram");
    return;
  }

  // Create histogram bins
  const bins = 20;
  const min = Math.min(...rulData);
  const max = Math.max(...rulData);
  const binSize = (max - min) / bins;

  const histogram = Array(bins).fill(0);
  const labels = [];

  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binSize;
    const binEnd = binStart + binSize;
    labels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}`);

    for (const value of rulData) {
      if (
        value >= binStart &&
        (value < binEnd || (i === bins - 1 && value === binEnd))
      ) {
        histogram[i]++;
      }
    }
  }

  rulHistogramChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Frequency",
          data: histogram,
          backgroundColor: "#3b82f6",
          borderColor: "#2563eb",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const total = rulData.length;
              const percentage =
                total > 0 ? ((context.parsed.y / total) * 100).toFixed(1) : 0;
              return `Count: ${context.parsed.y} (${percentage}%)`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Frequency",
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            precision: 0,
          },
        },
        x: {
          title: {
            display: true,
            text: "RUL Range (minutes)",
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            maxRotation: 45,
            minRotation: 45,
          },
        },
      },
    },
  });
}

function createUrgencyChart(statistics) {
  const ctx = document.getElementById("urgencyChart").getContext("2d");

  if (urgencyChart) {
    urgencyChart.destroy();
  }

  const healthyCount = statistics.healthy_count || 0;
  const warningCount = statistics.warning_count || 0;
  const severeCount = statistics.severe_count || 0;
  const criticalCount = statistics.critical_count || 0;
  const total = statistics.total_sequences || 1;

  urgencyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Low", "Medium", "High", "Critical"],
      datasets: [
        {
          label: "Count",
          data: [healthyCount, warningCount, severeCount, criticalCount],
          backgroundColor: ["#22c55e", "#f59e0b", "#ff6b35", "#ef4444"],
          borderColor: ["#16a34a", "#d97706", "#e55b2d", "#dc2626"],
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const value = context.parsed.y;
              const percentage = ((value / total) * 100).toFixed(1);
              return `Count: ${value} (${percentage}%)`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Number of Sequences",
            font: {
              size: 14,
              weight: "bold",
            },
          },
          ticks: {
            precision: 0,
          },
        },
        x: {
          title: {
            display: true,
            text: "Maintenance Urgency",
            font: {
              size: 14,
              weight: "bold",
            },
          },
        },
      },
    },
  });
}

// ==================== ALERT SYSTEM ====================

function showAlert(type, message) {
  const alertBox = document.getElementById("alertBox");
  const alertIcon = document.getElementById("alertIcon");
  const alertMessage = document.getElementById("alertMessage");

  alertBox.classList.remove("alert-success", "alert-error", "alert-warning");

  if (type === "success") {
    alertIcon.textContent = "✓";
    alertBox.classList.add("alert-success");
  } else if (type === "error") {
    alertIcon.textContent = "✗";
    alertBox.classList.add("alert-error");
  } else if (type === "warning") {
    alertIcon.textContent = "⚠️";
    alertBox.classList.add("alert-warning");
  }

  alertMessage.textContent = message;
  alertBox.style.display = "flex";

  setTimeout(() => {
    closeAlert();
  }, 5000);
}

function closeAlert() {
  document.getElementById("alertBox").style.display = "none";
}
