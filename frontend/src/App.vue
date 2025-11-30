<template>
  <div id="app">
    <nav class="main-nav">
      <div class="nav-container">
        <router-link to="/" class="nav-brand">Client Analytics</router-link>
        <div class="nav-links">
          <router-link to="/" class="nav-link">Дашборд</router-link>
          <router-link to="/analytics" class="nav-link">Аналитика</router-link>
        </div>
      </div>
    </nav>
    <main class="main-content">
      <div class="container">
        <!-- Header -->
        <div class="header">
          <div class="menu">меню</div>
          <div class="search-container">
            <button 
              class="refresh-button" 
              @click="refreshModelStatus"
              :disabled="refreshingStatus"
            >
              {{ refreshingStatus ? 'обновление...' : 'обновить статус' }}
            </button>
          </div>
        </div>
        
        <!-- Loading State -->
        <div v-if="loading" class="loading-state">
          <div class="loading-spinner"></div>
          <div>Анализ данных клиента...</div>
        </div>
        
        <!-- Error State -->
        <div v-if="error" class="error-state">
          <div class="error-message">{{ error }}</div>
          <button @click="clearError" class="retry-button">Попробовать снова</button>
        </div>
        
        <!-- Stats Section -->
        <div v-if="!loading && !error && currentClient" class="stats-section">
          <div class="profile-card">
            <div class="profile-value">{{ formatCurrency(clientData.incomePerClient) }}</div>
            <div class="profile-label">прогнозируемый доход</div>
            <div class="profile-accuracy">Точность: {{ staticAccuracy }}%</div>
          </div>
          <div class="stats-cards">
            <div class="stat-card">
              <div class="stat-value">{{ formatCurrency(clientData.currentRevenue) }}</div>
              <div class="stat-description">текущий доход</div>
            </div>
            <div class="stat-card">
              <div class="stat-segment">{{ clientData.segment }}</div>
              <div class="stat-description">сегмент клиента</div>
            </div>
          </div>
        </div>
        
        <!-- Default State when no client selected -->
        <div v-if="!loading && !error && !currentClient" class="welcome-state">
          <div class="welcome-message">
            Введите данные клиента для анализа дохода
          </div>
        </div>

        <!-- Client Input Form -->
        <div class="client-form-section">
          <div class="section-title">Данные клиента для анализа</div>
          <div class="client-form">
            <div class="form-row">
              <div class="input-group">
                <label>ID клиента:</label>
                <input v-model="clientInput.id" type="number" class="form-input">
              </div>
              <div class="input-group">
                <label>Возраст:</label>
                <input v-model="clientInput.age" type="number" class="form-input">
              </div>
            </div>
            <div class="form-row">
              <div class="input-group">
                <label>Доход (incomeValue):</label>
                <input v-model="clientInput.incomeValue" type="number" class="form-input">
              </div>
              <div class="input-group">
                <label>Обороты (turn_cur_cr_avg_act_v2):</label>
                <input v-model="clientInput.turn_cur_cr_avg_act_v2" type="number" class="form-input">
              </div>
            </div>
            <div class="form-row">
              <div class="input-group">
                <label>Лимит (hdb_bki_total_max_limit):</label>
                <input v-model="clientInput.hdb_bki_total_max_limit" type="number" class="form-input">
              </div>
              <div class="input-group">
                <label>Зарплата (salary_6to12m_avg):</label>
                <input v-model="clientInput.salary_6to12m_avg" type="number" class="form-input">
              </div>
            </div>
            <button 
              class="predict-button" 
              @click="analyzeClient"
              :disabled="makingPrediction"
            >
              {{ makingPrediction ? 'анализ...' : 'проанализировать доход' }}
            </button>
          </div>
        </div>
        
        <!-- Products Section -->
        <div v-if="!loading && !error && currentClient" class="products-section">
          <div class="section-title">Рекомендуемые продукты</div>
          <div class="products-grid">
            <div 
              v-for="(product, index) in clientData.recommendedProducts" 
              :key="index" 
              class="product-card"
            >
              <div class="product-title">{{ product.title }}</div>
              <div class="product-description">{{ product.description }}</div>
              <button 
                class="offer-button" 
                @click="offerProduct(product)"
                :disabled="product.offering"
              >
                {{ product.offering ? 'предлагается...' : 'предложить' }}
              </button>
            </div>
          </div>
        </div>
        
        <!-- Factors Section -->
        <div v-if="!loading && !error && currentClient" class="factors-section">
          <div class="section-title">Факторы влияния</div>
          <div class="factors-list">
            <div 
              v-for="(factor, index) in clientData.influenceFactors" 
              :key="index" 
              class="factor-item"
            >
              {{ factor }}
            </div>
          </div>
        </div>
        
        <!-- Monitoring Section -->
        <div class="monitoring-section">
          <div class="section-title">Мониторинг качества модели</div>
          <div class="monitoring-card">
            <div class="metrics-grid">
              <div class="metric-item">
                <div class="metric-value">{{ formatWMAE(staticWMAE) }}</div>
                <div class="metric-label">Текущий WMAE</div>
              </div>
              <div class="metric-item">
                <div class="metric-value">{{ modelMetrics.processedClients.toLocaleString() }}</div>
                <div class="metric-label">Обработано клиентов</div>
              </div>
              <div class="metric-item">
                <div class="metric-value">{{ staticAccuracy }}%</div>
                <div class="metric-label">Точность прогнозов</div>
              </div>
              <div class="metric-item">
                <div class="metric-value">{{ modelMetrics.activeModels }}</div>
                <div class="metric-label">Активных моделей</div>
              </div>
            </div>
            <div class="model-status-info">
              <div class="status-item" :class="{ 'active': modelStatus.model_loaded }">
                Модель: {{ modelStatus.model_loaded ? 'загружена' : 'не загружена' }}
              </div>
              <div class="status-item">
                Последнее обновление: {{ lastRefreshTime }}
              </div>
            </div>
          </div>
        </div>

        <!-- Model Info Section -->
        <div class="model-info-section">
          <div class="section-title">Информация о модели</div>
          <div class="info-card">
            <div class="info-item">
              <strong>Всего фич в полной модели:</strong> {{ featuresInfo.full_model_features_count }}
            </div>
            <div class="info-item">
              <strong>Фичи упрощенной модели:</strong> {{ featuresInfo.simple_model_features.join(', ') }}
            </div>
            <div class="info-item">
              <strong>Точность модели:</strong> {{ staticAccuracy }}%
            </div>
            <div class="info-item">
              <strong>WMAE модели:</strong> {{ formatWMAE(staticWMAE) }}
            </div>
            <div class="features-description">
              <div v-for="(desc, feature) in featuresInfo.simple_features_description" 
                   :key="feature" class="feature-desc">
                <strong>{{ feature }}:</strong> {{ desc }}
              </div>
            </div>
            <div class="model-actions">
              <button 
                class="refresh-button" 
                @click="refreshModelStatus"
                :disabled="refreshingStatus"
              >
                {{ refreshingStatus ? 'обновление...' : 'обновить статус модели' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
const apiService = {
  baseURL: 'http://localhost:8000',

  async getFeaturesInfo() {
    const response = await fetch(`${this.baseURL}/features/info`);
    if (!response.ok) {
      throw new Error(`Ошибка получения информации о фичах: ${response.status}`);
    }
    return await response.json();
  },

  async predictSingle(data) {
    const response = await fetch(`${this.baseURL}/predict_single`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      throw new Error(`Ошибка предсказания: ${response.status}`);
    }
    
    return await response.json();
  },

  async getModelMetrics() {
    const response = await fetch(`${this.baseURL}/model/metrics`);
    if (!response.ok) {
      throw new Error(`Ошибка получения метрик: ${response.status}`);
    }
    return await response.json();
  },

  async refreshModel() {
    const response = await fetch(`${this.baseURL}/refresh_model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`Ошибка обновления модели: ${response.status}`);
    }
    
    return await response.json();
  },

  async offerProduct(clientId, product) {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({ success: true, message: `Продукт "${product.title}" предложен клиенту ${clientId}` });
      }, 1000);
    });
  }
};

export default {
  name: 'App',
  
  data() {
    return {
      loading: false,
      error: null,
      makingPrediction: false,
      refreshingStatus: false,
      currentClient: null,
      // Статические метрики модели
      staticWMAE: 45215.087012,
      staticAccuracy: 94.2,
      lastRefreshTime: 'никогда',
      clientInput: {
        id: 1,
        age: 35,
        incomeValue: 50000,
        turn_cur_cr_avg_act_v2: 100000,
        hdb_bki_total_max_limit: 50000,
        salary_6to12m_avg: 45000
      },
      clientData: {
        incomePerClient: 0,
        currentRevenue: 0,
        revenueComparison: '',
        segment: '',
        recommendedProducts: [
          {
            id: 1,
            title: 'Кредитная карта Platinum',
            description: 'Персональный финансовый советник, премиальные инвестиционные продукты, эксклюзивные условия по кредитам',
            offering: false
          },
          {
            id: 2,
            title: 'Private Banking',
            description: 'Персональный финансовый советник, премиальные инвестиционные продукты, эксклюзивные условия по кредитам',
            offering: false
          },
          {
            id: 3,
            title: 'Инвестиционный портфель',
            description: 'Персональный финансовый советник, премиальные инвестиционные продукты, эксклюзивные условия по кредитам',
            offering: false
          }
        ],
        influenceFactors: []
      },
      modelStatus: {
        model_loaded: false,
        model_exists: false,
        wmae: null,
        features_count: 0
      },
      modelMetrics: {
        processedClients: 0,
        activeModels: 1
      },
      featuresInfo: {
        full_model_features_count: 0,
        simple_model_features: [],
        simple_features_description: {}
      }
    };
  },

  async mounted() {
    await this.loadFeaturesInfo();
    await this.loadModelMetrics();
    await this.loadModelStatus();
    this.updateRefreshTime();
  },

  methods: {
    async loadFeaturesInfo() {
      try {
        const featuresInfo = await apiService.getFeaturesInfo();
        this.featuresInfo = featuresInfo;
      } catch (error) {
        console.error('Ошибка при загрузке информации о фичах:', error);
        // Значения по умолчанию
        this.featuresInfo = {
          full_model_features_count: 156,
          simple_model_features: ['incomeValue', 'age', 'turn_cur_cr_avg_act_v2', 'hdb_bki_total_max_limit', 'salary_6to12m_avg'],
          simple_features_description: {
            "incomeValue": "Декларируемый доход клиента",
            "age": "Возраст клиента", 
            "turn_cur_cr_avg_act_v2": "Средние обороты по текущим кредитам",
            "hdb_bki_total_max_limit": "Общий максимальный лимит по БКИ",
            "salary_6to12m_avg": "Средняя зарплата за последние 6-12 месяцев"
          }
        };
      }
    },

    async loadModelMetrics() {
      try {
        const metrics = await apiService.getModelMetrics();
        this.modelMetrics.processedClients = metrics.processed_clients;
        this.modelMetrics.activeModels = metrics.active_models;
      } catch (error) {
        console.error('Ошибка при загрузке метрик модели:', error);
      }
    },

    async loadModelStatus() {
      try {
        const response = await fetch(`${apiService.baseURL}/model/status`);
        if (response.ok) {
          const status = await response.json();
          this.modelStatus = status;
        }
      } catch (error) {
        console.error('Ошибка при загрузке статуса модели:', error);
      }
    },

    async refreshModelStatus() {
      this.refreshingStatus = true;
      try {
        // Обновляем статус модели
        const refreshResult = await apiService.refreshModel();
        
        // Перезагружаем все данные
        await this.loadModelMetrics();
        await this.loadModelStatus();
        await this.loadFeaturesInfo();
        
        this.updateRefreshTime();
        
        console.log('Статус модели обновлен:', refreshResult);
        
      } catch (error) {
        console.error('Ошибка при обновлении статуса модели:', error);
        this.error = `Ошибка при обновлении статуса: ${error.message}`;
      } finally {
        this.refreshingStatus = false;
      }
    },

    updateRefreshTime() {
      this.lastRefreshTime = new Date().toLocaleString('ru-RU');
    },

    async analyzeClient() {
      this.makingPrediction = true;
      this.loading = true;
      this.error = null;

      try {
        const prediction = await apiService.predictSingle(this.clientInput);
        
        this.currentClient = this.clientInput.id;
        this.clientData.incomePerClient = prediction.prediction;
        this.clientData.currentRevenue = prediction.prediction;
        this.clientData.revenueComparison = this.generateRevenueComparison(prediction.prediction);
        this.clientData.segment = prediction.segment;
        this.clientData.influenceFactors = this.generateInfluenceFactors(prediction.prediction);

        await this.loadModelMetrics();

      } catch (error) {
        console.error('Ошибка при анализе клиента:', error);
        this.error = `Не удалось проанализировать клиента: ${error.message}`;
        this.currentClient = null;
      } finally {
        this.makingPrediction = false;
        this.loading = false;
      }
    },

    async offerProduct(product) {
      if (!this.currentClient) {
        alert('Пожалуйста, сначала введите данные клиента');
        return;
      }

      product.offering = true;

      try {
        const result = await apiService.offerProduct(this.currentClient, product);
        if (result.success) {
          alert(result.message);
        } else {
          alert('Ошибка при предложении продукта');
        }
      } catch (error) {
        console.error('Ошибка при предложении продукта:', error);
        alert('Ошибка при предложении продукта');
      } finally {
        product.offering = false;
      }
    },

    generateRevenueComparison(income) {
      const variations = [
        'На 13.9% выше предыдущего периода',
        'На 5.2% ниже предыдущего периода', 
        'Стабильный рост на 8.1%',
        'На 2.3% выше среднего по сегменту'
      ];
      return variations[Math.floor(Math.random() * variations.length)];
    },

    generateInfluenceFactors(income) {
      const baseFactors = [
        'Высокая средняя заработная плата',
        'Крупные кредитовые обороты', 
        'Низкая активность в мобильном банке',
        'Стабильные остатки на счетах',
        'Регулярные инвестиционные операции',
        'Высокий кредитный рейтинг'
      ];
      
      const factorCount = income > 80000 ? 4 : 3;
      return baseFactors
        .sort(() => Math.random() - 0.5)
        .slice(0, factorCount);
    },

    formatCurrency(value) {
      return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'RUB',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(value);
    },

    formatWMAE(value) {
      // Форматируем WMAE с округлением до 2 знаков после запятой
      return Number(value).toFixed(2);
    },

    clearError() {
      this.error = null;
    }
  }
}
</script>

<style>
{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Etude Noire', sans-serif;
  font-size: 14px;
  background: linear-gradient(rgba(248,248,248,1), rgba(210,215,225,1));
  min-height: 100vh;
  color: #000;
}

#app {
  min-height: 100vh;
}

.main-nav {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 1rem 0;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}

.nav-brand {
  font-size: 24px;
  font-weight: bold;
  color: rgba(213,72,104,1);
  text-decoration: none;
}

.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-link {
  text-decoration: none;
  color: #333;
  font-size: 16px;
  transition: color 0.3s ease;
}

.nav-link:hover,
.nav-link.router-link-active {
  color: rgba(213,72,104,1);
}

.main-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
}

/* Header section */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
}

.menu {
  color: white;
  font-size: 26px;
  background: rgba(213,72,104,1);
  padding: 10px 20px;
  border-radius: 16px;
  box-shadow: 0px 4px 6px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
}

.refresh-button {
  background: rgba(86,97,119,1);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 12px 24px;
  font-family: 'Etude Noire', sans-serif;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0px 4px 6px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
}

.refresh-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0px 6px 8px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
}

.refresh-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

/* Client Form Section */
.client-form-section {
  margin-bottom: 40px;
}

.client-form {
  background: rgba(240,233,240,1);
  border-radius: 16px;
  box-shadow: 0px 4px 6px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
  padding: 30px;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}

.input-group {
  display: flex;
  flex-direction: column;
}

.input-group label {
  margin-bottom: 8px;
  font-weight: 500;
  color: #333;
}

.form-input {
  padding: 12px 15px;
  border: 2px solid #e1e1e1;
  border-radius: 8px;
  font-family: 'Etude Noire', sans-serif;
  font-size: 14px;
  transition: border-color 0.3s ease;
  background: white;
}

.form-input:focus {
  outline: none;
  border-color: rgba(213,72,104,1);
}

.predict-button {
  background: rgba(86,97,119,1);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 15px 30px;
  font-family: 'Etude Noire', sans-serif;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-top: 10px;
  width: 100%;
}

.predict-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0px 4px 6px rgba(0,0,0,0.15);
}

.predict-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

/* Новые стили для статической точности */
.profile-accuracy {
  font-size: 14px;
  margin-top: 10px;
  opacity: 0.9;
}

.model-status-info {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid rgba(0,0,0,0.1);
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.status-item {
  padding: 10px 15px;
  background: rgba(255,255,255,0.7);
  border-radius: 8px;
  font-size: 14px;
}

.status-item.active {
  background: rgba(76, 175, 80, 0.2);
  color: #2e7d32;
}

.model-actions {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid rgba(0,0,0,0.1);
  display: flex;
  justify-content: center;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
  gap: 20px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid rgba(213,72,104,1);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
  gap: 20px;
  background: rgba(255, 235, 235, 0.8);
  border-radius: 16px;
  margin: 20px 0;
}

.error-message {
  color: #d32f2f;
  text-align: center;
  font-size: 16px;
}

.retry-button {
  background: rgba(213,72,104,1);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 10px 20px;
  font-family: 'Etude Noire', sans-serif;
  cursor: pointer;
  transition: all 0.3s ease;
}

.retry-button:hover {
  transform: translateY(-2px);
  box-shadow: 0px 4px 6px rgba(0,0,0,0.15);
}

.welcome-state {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 60px 20px;
  text-align: center;
}

.welcome-message {
  font-size: 18px;
  color: rgba(0,0,0,0.6);
  background: rgba(240,233,240,1);
  padding: 30px;
  border-radius: 16px;
  box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
}

.stats-section {
  display: flex;
  gap: 20px;
  margin-bottom: 40px;
}

.profile-card {
  width: 305px;
  height: 305px;
  background: linear-gradient(rgba(213,72,104,1), rgba(240,233,240,1));
  border-radius: 50%;
  box-shadow: 0px 6px 8px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: white;
}

.profile-value {
  font-size: 36px;
  margin-bottom: 10px;
}

.profile-label {
  font-size: 18px;
}

.stats-cards {
  display: flex;
  flex: 1;
  gap: 20px;
  align-items: center;
}

.stat-card {
  flex: 1;
  background: rgba(240,233,240,1);
  border-radius: 16px;
  box-shadow: 0px 4px 6px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
  padding: 10px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 100px;
  max-height: 120px;
}

.stat-value {
  font-size: 24px;
  margin-bottom: 5px;
}

.stat-description {
  font-size: 12px;
  color: rgba(0,0,0,0.7);
}

.stat-segment {
  font-size: 20px;
}

.products-section {
  margin-bottom: 40px;
}

.section-title {
  font-size: 24px;
  margin-bottom: 15px;
}

.products-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.product-card {
  background: rgba(240,233,240,1);
  border-radius: 16px;
  box-shadow: 0px 4px 6px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
  padding: 20px;
  display: flex;
  flex-direction: column;
}

.product-title {
  font-size: 14px;
  margin-bottom: 10px;
}

.product-description {
  font-size: 12px;
  margin-bottom: 20px;
  flex-grow: 1;
}

.offer-button {
  background: rgba(213,72,104,1);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 10px 20px;
  font-family: 'Etude Noire', sans-serif;
  font-size: 15px;
  cursor: pointer;
  transition: all 0.3s ease;
  align-self: flex-start;
  box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
}

.offer-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0px 4px 6px rgba(0,0,0,0.15);
}

.offer-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.factors-section {
  margin-bottom: 40px;
}

.factors-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.factor-item {
  background: rgba(240,233,240,1);
  border: 5px solid rgba(86,97,119,1);
  border-radius: 16px;
  padding: 20px;
  font-size: 14px;
  transition: all 0.3s ease;
}

.factor-item:hover {
  transform: translateY(-2px);
  box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
}

.monitoring-section {
  margin-bottom: 40px;
}

.monitoring-card {
  background: rgba(240,233,240,1);
  border-radius: 16px;
  box-shadow: 0px 4px 6px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
  padding: 30px;
  min-height: 166px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.metric-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 15px;
  background: rgba(255,255,255,0.7);
  border-radius: 12px;
  box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
}

.metric-value {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 5px;
  color: rgba(213,72,104,1);
}

.metric-label {
  font-size: 14px;
  color: rgba(0,0,0,0.7);
}

.model-info-section {
  margin-bottom: 40px;
}

.info-card {
  background: rgba(240,233,240,1);
  border-radius: 16px;
  box-shadow: 0px 4px 6px rgba(0.33725491166114807, 0.3803921639919281, 0.46666666865348816, 0.5);
  padding: 30px;
}

.info-item {
  margin-bottom: 15px;
  font-size: 16px;
}

.features-description {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid rgba(0,0,0,0.1);
}

.feature-desc {
  margin-bottom: 10px;
  font-size: 14px;
  color: rgba(0,0,0,0.8);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .stats-section {
    flex-direction: column;
  }
  
  .profile-card {
    align-self: center;
  }
  
  .stats-cards {
    flex-direction: column;
  }
  
  .header {
    flex-direction: column;
    gap: 15px;
  }
  
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .form-row {
    grid-template-columns: 1fr;
    gap: 15px;
  }
  
  .model-status-info {
    flex-direction: column;
    gap: 10px;
  }
}

@media (max-width: 480px) {
  .metrics-grid {
    grid-template-columns: 1fr;
  }
}
</style>