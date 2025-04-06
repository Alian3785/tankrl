import pygame
import sys
import math
import random
import numpy as np # Обязательно для Gymnasium

import gymnasium as gym
from gymnasium import spaces

# --- Константы Игры ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

TANK_WIDTH = 40
TANK_HEIGHT = 50
TANK_SPEED = 5 # Немного ускорим для RL
ENEMY_SPEED = 3
PROJECTILE_SPEED = TANK_SPEED * 2
PROJECTILE_RADIUS = 5

FPS = 60 # Для рендеринга, шаг среды не привязан к FPS напрямую

ENEMY_CHANGE_DIR_TIME = 1.5 * FPS # В секундах, пересчитаем в шагах среды позже
ENEMY_SHOOT_TIME = 3.0 * FPS      # В секундах

MAX_PLAYER_PROJECTILES = 10 # Ограничение для observation space
MAX_ENEMY_PROJECTILES = 15  # Ограничение для observation space
MAX_ENEMIES = 3

# --- Игровые Классы (Почти без изменений, но убрана обработка keys в PlayerTank) ---

# --- Класс Танка (Общий, используется как база) ---
class BaseTank(pygame.sprite.Sprite):
    def __init__(self, x, y, color, speed, img_width=TANK_WIDTH, img_height=TANK_HEIGHT):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.original_image = pygame.Surface([img_width, img_height], pygame.SRCALPHA)
        # Общий стиль танка
        pygame.draw.rect(self.original_image, color, (0, 10, img_width, img_height - 10)) # Корпус
        pygame.draw.rect(self.original_image, color, (img_width // 2 - 5, 0, 10, img_height // 2)) # Ствол (вверх)
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = speed
        self.x = float(self.rect.centerx)
        self.y = float(self.rect.centery)
        self.direction_vector = pygame.math.Vector2(0, -1) # Начинает смотреть вверх
        self.angle = 0
        self.rotate() # Начальный поворот

    def update(self, move_vector=None): # Принимает вектор движения или None
        if not self.alive(): return

        # Движение, если задан вектор
        if move_vector and move_vector.length() > 0:
            # Обновляем направление только если движемся
            if move_vector != self.direction_vector:
                 # Важно: нормализуем вектор, если это просто направление
                 # Если это уже вектор смещения, то используем его как есть
                 # В текущей реализации step передает нормализованный вектор
                 self.direction_vector = move_vector.normalize()
                 self.rotate()

            # Применяем движение
            self.x += move_vector.x * self.speed
            self.y += move_vector.y * self.speed
            self.apply_bounds()
            self.rect.center = (int(self.x), int(self.y))


    def rotate(self):
        self.angle = math.degrees(math.atan2(-self.direction_vector.y, self.direction_vector.x)) - 90
        old_center = self.rect.center
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=old_center)

    def apply_bounds(self):
        half_w = self.rect.width / 2
        half_h = self.rect.height / 2
        # Используем глобальные переменные SCREEN_WIDTH, SCREEN_HEIGHT
        self.x = max(half_w, min(self.x, SCREEN_WIDTH - half_w))
        self.y = max(half_h, min(self.y, SCREEN_HEIGHT - half_h))

    def shoot(self, projectile_color):
        offset_distance = self.img_height / 2
        spawn_pos = pygame.math.Vector2(self.rect.center) + self.direction_vector * offset_distance
        return Projectile(spawn_pos.x, spawn_pos.y, self.direction_vector.copy(), projectile_color)

# --- Класс Танка Игрока ---
class PlayerTank(BaseTank):
    def __init__(self, x, y):
        super().__init__(x, y, GREEN, TANK_SPEED)

    # Update игрока теперь принимает action, а не keys
    def update(self, action): # action: 0:up, 1:down, 2:left, 3:right, 4:shoot, 5: no-op
        if not self.alive(): return

        move_vector = pygame.math.Vector2(0, 0)
        shooting = False

        if action == 0: # Вверх
            move_vector.y = -1
        elif action == 1: # Вниз
            move_vector.y = 1
        elif action == 2: # Влево
            move_vector.x = -1
        elif action == 3: # Вправо
            move_vector.x = 1
        elif action == 4: # Выстрел
            shooting = True
        # action == 5 (no-op) -> ничего не делаем

        # Вызываем update базового класса для движения
        if move_vector.length() > 0:
            super().update(move_vector=move_vector)
        else: # Если нет движения, все равно вызываем для возможной логики (хотя сейчас ее нет)
             super().update(move_vector=None)

        return shooting # Возвращаем, хочет ли танк стрелять

# --- Класс Вражеского Танка ---
class EnemyTank(BaseTank):
    def __init__(self, x, y, change_dir_interval, shoot_interval):
        super().__init__(x, y, BLUE, ENEMY_SPEED)
        # Интервалы в шагах среды (не кадрах pygame)
        self.change_dir_interval = change_dir_interval
        self.shoot_interval = shoot_interval
        self.move_timer = random.randint(1, self.change_dir_interval)
        self.shoot_timer = random.randint(1, self.shoot_interval)
        self.wants_to_shoot = False # Флаг для среды

    # Update врага не зависит от action агента
    def update(self, *args): # Принимает *args для совместимости с all_sprites.update
        if not self.alive(): return

        # --- Логика движения ---
        self.move_timer -= 1
        if self.move_timer <= 0:
            self.direction_vector = random.choice([
                pygame.math.Vector2(0, 1), pygame.math.Vector2(0, -1),
                pygame.math.Vector2(1, 0), pygame.math.Vector2(-1, 0)
            ])
            self.rotate()
            self.move_timer = self.change_dir_interval + random.randint(-self.change_dir_interval//4, self.change_dir_interval//4)

        # Вызываем update базового класса для движения
        super().update(move_vector=self.direction_vector) # Всегда движется

        # --- Логика стрельбы ---
        self.shoot_timer -= 1
        self.wants_to_shoot = False # Сбрасываем флаг каждый шаг
        if self.shoot_timer <= 0:
            self.wants_to_shoot = True # Устанавливаем флаг для среды
            self.shoot_timer = self.shoot_interval + random.randint(-self.shoot_interval//2, self.shoot_interval//2)


# --- Класс Проджектайла ---
class Projectile(pygame.sprite.Sprite):
    def __init__(self, x, y, direction_vector, color):
        super().__init__()
        self.image = pygame.Surface([PROJECTILE_RADIUS * 2, PROJECTILE_RADIUS * 2], pygame.SRCALPHA)
        pygame.draw.circle(self.image, color, (PROJECTILE_RADIUS, PROJECTILE_RADIUS), PROJECTILE_RADIUS)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = PROJECTILE_SPEED
        if direction_vector.length() > 0:
             self.direction_vector = direction_vector.normalize()
        else:
             self.direction_vector = pygame.math.Vector2(0, -1) # Безопасное значение
        self.x = float(self.rect.centerx)
        self.y = float(self.rect.centery)

    def update(self, *args): # Принимает *args для совместимости с all_sprites.update
        move_vec = self.direction_vector * self.speed
        self.x += move_vec.x
        self.y += move_vec.y
        self.rect.center = (int(self.x), int(self.y))

        # Удаление при выходе за экран (используем константы)
        if (self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT or
                self.rect.right < 0 or self.rect.left > SCREEN_WIDTH):
            self.kill()

# --- Gymnasium Среда ---
class TankEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode="human"):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Tank RL Environment")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        elif self.render_mode == "rgb_array":
             # Инициализация Pygame нужна для поверхностей и шрифтов, но без экрана
             pygame.init()
             # Создаем 'виртуальный' экран для рендеринга в массив
             self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
             # Clock не нужен для rgb_array, так как шаг не привязан к реальному времени

        # --- Пространство Действий ---
        # 0: Вверх, 1: Вниз, 2: Влево, 3: Вправо, 4: Выстрел, 5: Ничего не делать (No-op)
        self.action_space = spaces.Discrete(6)

        # --- Пространство Наблюдений ---
        # Координаты (x, y) нормализованы к [0, 1]
        # [player_x, player_y,
        #  enemy1_x, enemy1_y, enemy1_alive, # Добавляем флаг жив ли враг
        #  enemy2_x, enemy2_y, enemy2_alive,
        #  enemy3_x, enemy3_y, enemy3_alive,
        #  proj_p1_x, proj_p1_y, ..., proj_pN_x, proj_pN_y, (MAX_PLAYER_PROJECTILES * 2)
        #  proj_e1_x, proj_e1_y, ..., proj_eM_x, proj_eM_y] (MAX_ENEMY_PROJECTILES * 2)
        # Используем -1 для отсутствующих объектов или координат
        obs_size = 2 + MAX_ENEMIES * 3 + MAX_PLAYER_PROJECTILES * 2 + MAX_ENEMY_PROJECTILES * 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Пересчитываем интервалы AI врагов в шаги (если FPS используется для рендеринга)
        # Для RL лучше иметь фиксированные интервалы в шагах
        self.enemy_change_dir_interval = 60 # Примерно каждые 60 шагов
        self.enemy_shoot_interval = 150     # Примерно каждые 150 шагов

        # Состояние игры (инициализируется в reset)
        self.player_tank = None
        self.enemy_tanks = pygame.sprite.Group()
        self.player_projectiles = pygame.sprite.Group()
        self.enemy_projectiles = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.game_over = False
        self.win = False

        # Для хранения ссылок на врагов, чтобы заполнять observation space в том же порядке
        self._enemy_list = []

    def _get_obs(self):
        obs = np.full(self.observation_space.shape, -1.0, dtype=np.float32) # Заполняем -1

        # Координаты игрока
        if self.player_tank.alive():
            obs[0] = self.player_tank.x / SCREEN_WIDTH * 2 - 1 # Нормализация в [-1, 1]
            obs[1] = self.player_tank.y / SCREEN_HEIGHT * 2 - 1
        # else: оставляем -1

        # Координаты и состояние врагов
        obs_idx = 2
        for i in range(MAX_ENEMIES):
            if i < len(self._enemy_list) and self._enemy_list[i].alive():
                enemy = self._enemy_list[i]
                obs[obs_idx] = enemy.x / SCREEN_WIDTH * 2 - 1
                obs[obs_idx + 1] = enemy.y / SCREEN_HEIGHT * 2 - 1
                obs[obs_idx + 2] = 1.0 # Жив
            else:
                # obs[obs_idx] = -1.0 (уже есть)
                # obs[obs_idx + 1] = -1.0 (уже есть)
                obs[obs_idx + 2] = 0.0 # Мертв или отсутствует
            obs_idx += 3

        # Координаты снарядов игрока
        for i, proj in enumerate(self.player_projectiles):
            if i >= MAX_PLAYER_PROJECTILES: break
            obs[obs_idx] = proj.x / SCREEN_WIDTH * 2 - 1
            obs[obs_idx + 1] = proj.y / SCREEN_HEIGHT * 2 - 1
            obs_idx += 2
        obs_idx = 2 + MAX_ENEMIES * 3 + MAX_PLAYER_PROJECTILES * 2 # Переходим к индексу снарядов врага

        # Координаты снарядов врагов
        for i, proj in enumerate(self.enemy_projectiles):
            if i >= MAX_ENEMY_PROJECTILES: break
            obs[obs_idx] = proj.x / SCREEN_WIDTH * 2 - 1
            obs[obs_idx + 1] = proj.y / SCREEN_HEIGHT * 2 - 1
            obs_idx += 2

        return obs

    def _get_info(self):
        # Можно добавить доп. информацию для отладки
        return {
            "player_pos": (self.player_tank.x, self.player_tank.y) if self.player_tank.alive() else None,
            "enemies_left": len(self.enemy_tanks),
            "player_projectiles": len(self.player_projectiles),
            "enemy_projectiles": len(self.enemy_projectiles),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Для воспроизводимости Gymnasium

        # Сброс состояния игры
        self.game_over = False
        self.win = False

        # Очистка групп спрайтов
        self.all_sprites.empty()
        self.enemy_tanks.empty()
        self.player_projectiles.empty()
        self.enemy_projectiles.empty()
        self._enemy_list = []

        # Создание игрока
        self.player_tank = PlayerTank(SCREEN_WIDTH // 2, SCREEN_HEIGHT - TANK_HEIGHT * 1.5)
        self.all_sprites.add(self.player_tank)

        # Создание врагов
        for i in range(MAX_ENEMIES):
            enemy_x = (i + 1) * SCREEN_WIDTH // (MAX_ENEMIES + 1)
            enemy_y = TANK_HEIGHT
            enemy = EnemyTank(enemy_x, enemy_y, self.enemy_change_dir_interval, self.enemy_shoot_interval)
            self.all_sprites.add(enemy)
            self.enemy_tanks.add(enemy)
            self._enemy_list.append(enemy) # Сохраняем порядок

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame() # Отрисовка начального состояния

        return observation, info

    def step(self, action):
        if self.game_over:
            # Если игра уже закончена, просто возвращаем последнее состояние
            # Это стандартное поведение для Gymnasium v26+
            return self._get_obs(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False # Игра закончилась (победа/поражение)
        truncated = False # Эпизод прерван (например, по времени - не используется здесь)

        # 1. Обновление танка игрока на основе действия
        player_wants_to_shoot = self.player_tank.update(action)
        if player_wants_to_shoot and len(self.player_projectiles) < MAX_PLAYER_PROJECTILES:
            new_projectile = self.player_tank.shoot(RED)
            self.all_sprites.add(new_projectile)
            self.player_projectiles.add(new_projectile)
            # reward -= 0.05 # Небольшой штраф за выстрел? (Опционально)

        # 2. Обновление врагов и их стрельба
        for enemy in self.enemy_tanks:
            enemy.update() # Враги обновляются сами
            if enemy.wants_to_shoot and len(self.enemy_projectiles) < MAX_ENEMY_PROJECTILES:
                new_projectile = enemy.shoot(YELLOW)
                self.all_sprites.add(new_projectile)
                self.enemy_projectiles.add(new_projectile)

        # 3. Обновление снарядов
        self.player_projectiles.update()
        self.enemy_projectiles.update()

        # 4. Проверка столкновений и начисление наград/штрафов
        # Снаряды игрока -> Враги
        enemies_hit = pygame.sprite.groupcollide(self.enemy_tanks, self.player_projectiles, True, True)
        if enemies_hit:
            reward += 15.0 * len(enemies_hit) # Награда за каждого сбитого врага
            # Проверяем условие победы сразу после уничтожения
            if len(self.enemy_tanks) == 0:
                self.win = True
                self.game_over = True
                terminated = True
                reward += 100.0 # Большая награда за победу

        # Снаряды врагов -> Игрок
        if not self.game_over: # Проверяем только если игра еще не закончена победой
            player_hits = pygame.sprite.spritecollide(self.player_tank, self.enemy_projectiles, True)
            if player_hits:
                reward -= 50.0 # Большой штраф за попадание
                self.player_tank.kill()
                self.game_over = True
                terminated = True
                self.win = False # Убедимся, что это не победа

        # Небольшой штраф за существование, чтобы побудить к действию
        reward -= 0.01

        # 5. Получение нового наблюдения и информации
        observation = self._get_obs()
        info = self._get_info()

        # 6. Рендеринг (если включен)
        if self.render_mode == "human":
            self._render_frame()

        # Важно: если игра закончилась (terminated=True), агент должен вызвать reset()
        return observation, reward, terminated, truncated, info


    def render(self):
        # render() теперь в основном возвращает массив для 'rgb_array'
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # Для 'human' рендеринг происходит внутри step

    def _render_frame(self):
        if self.screen is None:
            if self.render_mode == "human":
                 # Инициализация если не было сделано или окно закрыли
                 pygame.init()
                 pygame.display.set_caption("Tank RL Environment")
                 self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            elif self.render_mode == "rgb_array":
                 # Нужно для rgb_array, если pygame не был инициализирован
                 pygame.init()
                 self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))


        # --- Отрисовка ---
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen) # Рисуем все живые спрайты

        # Отображение сообщений при окончании игры (только в human режиме)
        if self.render_mode == "human" and self.game_over:
             font = pygame.font.Font(pygame.font.match_font('arial'), 64)
             if self.win:
                 text_surface = font.render("ПОБЕДА!", True, GREEN)
             else:
                 text_surface = font.render("ПОРАЖЕНИЕ!", True, RED)
             text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
             self.screen.blit(text_surface, text_rect)

        if self.render_mode == "human":
            pygame.event.pump() # Обработка внутренних событий pygame
            pygame.display.flip() # Обновление экрана
            if self.clock: self.clock.tick(self.metadata["render_fps"]) # Контроль FPS
        elif self.render_mode == "rgb_array":
            # Конвертация поверхности Pygame в массив numpy для Gym
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None # Важно для предотвращения повторной инициализации

# --- Пример использования среды ---
if __name__ == '__main__':
    # Создаем среду с рендерингом
    # env = TankEnv(render_mode="human")
    # или без рендеринга для быстрого обучения
    env = TankEnv()

    # Проверка среды Gymnasium (рекомендуется)
    from gymnasium.utils.env_checker import check_env
    try:
        check_env(env)
        print("Gymnasium environment check passed!")
    except Exception as e:
        print(f"Gymnasium environment check failed: {e}")


    # Пример простого цикла с случайными действиями
    episodes = 33
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not terminated and not truncated:
            action = env.action_space.sample() # Случайное действие
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Если используете human рендеринг, можно добавить небольшую паузу
        if env.render_mode == "human":
            import time
            time.sleep(0.01)

        print(f"Episode {episode}: Steps={steps}, Total Reward={total_reward:.2f}, Terminated={terminated}, Truncated={truncated}, Win={env.win}")

    env.close()