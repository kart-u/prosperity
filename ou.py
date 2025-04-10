import numpy as np
import math
from statistics import NormalDist
from datamodel import *
from typing import List, Dict, Tuple, Any
import json

INF = 1e9
normalDist = NormalDist(0,1)


def is_mean_reverting(prices: np.ndarray, window: int = 20) -> bool:
        """
        Test if a price series is mean-reverting using variance ratio test.
        Lower short-term to long-term variance ratio suggests mean reversion.
        """
        if len(prices) < 30:
            return False
            
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate variance ratios
        short_var = np.var(returns)
        long_var = np.var(np.sum(returns.reshape(-1, window), axis=1)) / window
        
        # Mean reversion indicated by variance ratio < 0.8
        return (short_var / long_var) < 0.8


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Status:
    kalman_mean = 0  # Initial estimate of mean
    kalman_variance = 1  # Initial estimate of variance (uncertainty)
    kalman_process_variance = 1e-5  # Small value: how much we expect the mean to move
    kalman_measurement_variance = 1e-2  # Noise in price measurements

    _position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50
    }

    _state = None

    _realtime_position = {key:0 for key in _position_limit.keys()}

   

    _hist_order_depths = {
        product:{
            'bidprc1': [],
            'bidamt1': [],
            'bidprc2': [],
            'bidamt2': [],
            'bidprc3': [],
            'bidamt3': [],
            'askprc1': [],
            'askamt1': [],
            'askprc2': [],
            'askamt2': [],
            'askprc3': [],
            # amt is volume
            'askamt3': [],
            'fair_prices':[],
        } for product in _position_limit.keys()
    }

    _hist_observation = {
        'bidPrice': [],
        'askPrice': [],
    }

    _num_data = 0

    def __init__(self, product: str) -> None:
        """Initialize status object.

        Args:
            product (str): product

        """
        self.product = product

    def update_kalman_mean(self, price: float):
        # Prediction update
        self.kalman_variance += self.kalman_process_variance

        # Measurement update
        kalman_gain = self.kalman_variance / (self.kalman_variance + self.kalman_measurement_variance)
        self.kalman_mean += kalman_gain * (price - self.kalman_mean)
        self.kalman_variance *= (1 - kalman_gain)

    @property
    def kalman_filtered_mean(self):
        return self.kalman_mean


    @classmethod
    def cls_update(cls, state: TradingState) -> None:
        """Update trading state.
        have historical data to formulate strategies

        Args:
            state (TradingState): trading state

        """
        # Update TradingState
        cls._state = state
        # Update realtime position
        for product, posit in state.position.items():
            cls._realtime_position[product] = posit
        # Update historical order_depths
        for product, orderdepth in state.order_depths.items():
            cnt = 1
            mx=0
            mn=1e9
            for prc, amt in sorted(orderdepth.sell_orders.items(), reverse=False):
                mx=max(mx,amt)
                cls._hist_order_depths[product][f'askamt{cnt}'].append(amt)
                cls._hist_order_depths[product][f'askprc{cnt}'].append(prc)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'askprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'askamt{cnt}'].append(np.nan)
                cnt += 1
            cnt = 1
            for prc, amt in sorted(orderdepth.buy_orders.items(), reverse=True):
                mn=min(mn,prc)
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(prc)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(amt)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(np.nan)
                cnt += 1
            cls._hist_order_depths[product]['fair_prices'].append((mx+mn)//2)
        cls._num_data += 1
        # cls._hist_observation['sunlight'].append(state.observations.conversionObservations['ORCHIDS'].sunlight)
        # cls._hist_observation['humidity'].append(state.observations.conversionObservations['ORCHIDS'].humidity)
        # cls._hist_observation['transportFees'].append(state.observations.conversionObservations['ORCHIDS'].transportFees)
        # cls._hist_observation['exportTariff'].append(state.observations.conversionObservations['ORCHIDS'].exportTariff)
        # cls._hist_observation['importTariff'].append(state.observations.conversionObservations['ORCHIDS'].importTariff)
        # cls._hist_observation['bidPrice'].append(state.observations.conversionObservations['ORCHIDS'].bidPrice)
        # cls._hist_observation['askPrice'].append(state.observations.conversionObservations['ORCHIDS'].askPrice)

    def hist_order_depth(self, type: str, depth: int, size) -> np.ndarray:
        """Return historical order depth.

        Args:
            type (str): 'bidprc' or 'bidamt' or 'askprc' or 'askamt'
            depth (int): depth, 1 or 2 or 3
            size (int): size of data /length of array

        Returns:
            np.ndarray: historical order depth for given type and depth

        """
        return np.array(self._hist_order_depths[self.product][f'{type}{depth}'][-size:], dtype=np.float32)
    
    @property
    def timestep(self) -> int:
        return self._state.timestamp / 100

    @property
    def position_limit(self) -> int:
        """Return position limit of product.

        Returns:
            int: position limit of product

        """
        return self._position_limit[self.product]

    @property
    def position(self) -> int:
        """Return current position of product.

        Returns:
            int: current position of product

        """
        if self.product in self._state.position:
            return int(self._state.position[self.product])
        else:
            return 0
    
    @property
    def rt_position(self) -> int:
        """Return realtime position.

        Returns:
            int: realtime position

        """
        return self._realtime_position[self.product]

    def _cls_rt_position_update(cls, product, new_position):
        if abs(new_position) <= cls._position_limit[product]:
            cls._realtime_position[product] = new_position
        else:
            raise ValueError("New position exceeds position limit")

    def rt_position_update(self, new_position: int) -> None:
        """Update realtime position.

        Args:
            new_position (int): new position

        """
        self._cls_rt_position_update(self.product, new_position)
    
    @property
    def bids(self) -> list[tuple[int, int]]:
        """Return bid orders.

        Returns:
            dict[int, int].items(): bid orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].buy_orders.items())
    
    @property
    def asks(self) -> list[tuple[int, int]]:
        """Return ask orders.

        Returns:
            dict[int, int].items(): ask orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].sell_orders.items())
    
    @classmethod
    def _cls_update_bids(cls, product, prc, new_amt):
        if new_amt > 0:
            cls._state.order_depths[product].buy_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].buy_orders[prc] = 0
        # else:
        #     raise ValueError("Negative amount in bid orders")

    @classmethod
    def _cls_update_asks(cls, product, prc, new_amt):
        if new_amt < 0:
            cls._state.order_depths[product].sell_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].sell_orders[prc] = 0
        # else:
        #     raise ValueError("Positive amount in ask orders")
        
    def update_bids(self, prc: int, new_amt: int) -> None:
        """Update bid orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_bids(self.product, prc, new_amt)
    
    def update_asks(self, prc: int, new_amt: int) -> None:
        """Update ask orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_asks(self.product, prc, new_amt)

    @property
    def possible_buy_amt(self) -> int:
        """Return possible buy amount.

        Returns:
            int: possible buy amount
        
        """
        possible_buy_amount1 = self._position_limit[self.product] - self.rt_position
        possible_buy_amount2 = self._position_limit[self.product] - self.position
        return min(possible_buy_amount1, possible_buy_amount2)
        
    @property
    def possible_sell_amt(self) -> int:
        """Return possible sell amount.

        Returns:
            int: possible sell amount
        
        """
        possible_sell_amount1 = self._position_limit[self.product] + self.rt_position
        possible_sell_amount2 = self._position_limit[self.product] + self.position
        return min(possible_sell_amount1, possible_sell_amount2)

    def hist_mid_prc(self, size:int) -> np.ndarray:
        """Return historical mid price.

        Args:
            size (int): size of data

        Returns:
            np.ndarray: historical mid price
        
        """
        return (self.hist_order_depth('bidprc', 1, size) + self.hist_order_depth('askprc', 1, size)) / 2
    
    def hist_maxamt_askprc(self, size:int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth
        
        """
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('askprc', 1, size), self.hist_order_depth('askprc', 2, size), self.hist_order_depth('askprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('askamt', 1, size), self.hist_order_depth('askamt', 2, size), self.hist_order_depth('askamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]

        return res_array

    def hist_maxamt_bidprc(self, size:int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth
        
        """
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('bidprc', 1, size), self.hist_order_depth('bidprc', 2, size), self.hist_order_depth('bidprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('bidamt', 1, size), self.hist_order_depth('bidamt', 2, size), self.hist_order_depth('bidamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]

        return res_array
    
    def hist_vwap_all(self, size:int) -> np.ndarray:
        res_array = np.zeros(size)
        volsum_array = np.zeros(size)
        for i in range(1,4):
            tmp_bid_vol = self.hist_order_depth(f'bidamt', i, size)
            tmp_ask_vol = self.hist_order_depth(f'askamt', i, size)
            tmp_bid_prc = self.hist_order_depth(f'bidprc', i, size)
            tmp_ask_prc = self.hist_order_depth(f'askprc', i, size)
            if i == 0:
                res_array = res_array + (tmp_bid_prc*tmp_bid_vol) + (-tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + tmp_bid_vol - tmp_ask_vol
            else:
                bid_nan_idx = np.isnan(tmp_bid_prc)
                ask_nan_idx = np.isnan(tmp_ask_prc)
                res_array = res_array + np.where(bid_nan_idx, 0, tmp_bid_prc*tmp_bid_vol) + np.where(ask_nan_idx, 0, -tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + np.where(bid_nan_idx, 0, tmp_bid_vol) - np.where(ask_nan_idx, 0, tmp_ask_vol)
                
        return res_array / volsum_array
    
    def hist_obs_bidPrice(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['bidPrice'][-size:], dtype=np.float32)
    
    def hist_obs_askPrice(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['askPrice'][-size:], dtype=np.float32)

    @property
    def best_bid(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return max(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def best_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return min(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def bid_ask_spread(self) -> int:
        return self.best_ask - self.best_bid

    @property
    def best_bid_amount(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        best_prc = max(self._state.order_depths[self.product].buy_orders.keys())
        best_amt = self._state.order_depths[self.product].buy_orders[best_prc]
        return best_amt
        
    @property
    def best_ask_amount(self) -> int:
        """Return best ask price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        best_prc = min(self._state.order_depths[self.product].sell_orders.keys())
        best_amt = self._state.order_depths[self.product].sell_orders[best_prc]
        return -best_amt
    
    @property
    def worst_bid(self) -> int:
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return min(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def worst_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return max(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def vwap(self) -> float:
        vwap = 0
        total_amt = 0

        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
            total_amt += amt

        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * abs(amt))
            total_amt += abs(amt)

        vwap /= total_amt
        return vwap

    @property
    def vwap_bidprc(self) -> float:
        """Return volume weighted average price of bid orders.

        Returns:
            float: volume weighted average price of bid orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
        vwap /= sum(self._state.order_depths[self.product].buy_orders.values())
        return vwap
    
    @property
    def vwap_askprc(self) -> float:
        """Return volume weighted average price of ask orders.

        Returns:
            float: volume weighted average price of ask orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * -amt)
        vwap /= -sum(self._state.order_depths[self.product].sell_orders.values())
        return vwap

    @property
    def maxamt_bidprc(self) -> int:
        """Return price of bid order with maximum amount.
        
        Returns:
            int: price of bid order with maximum amount

        """
        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            if amt > max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    
    @property
    def maxamt_askprc(self) -> int:
        """Return price of ask order with maximum amount.

        Returns:
            int: price of ask order with maximum amount
        
        """
        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            if amt < max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    
    @property
    def maxamt_midprc(self) -> float:
        return (self.maxamt_bidprc + self.maxamt_askprc) / 2

    def bidamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].buy_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0
        
    def askamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].sell_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0

    @property
    def total_bidamt(self) -> int:
        return sum(self._state.order_depths[self.product].buy_orders.values())

    @property
    def total_askamt(self) -> int:
        return -sum(self._state.order_depths[self.product].sell_orders.values())

    @property
    def market_trades(self) -> list:
        return self._state.market_trades.get(self.product, [])
    
    @classmethod
    def _cls_fair(cls,product)->List:
        return cls._hist_order_depths[product]['fair_prices']
    

    @property 
    def fair_price(self)->List:
        return self._cls_fair(self.product)


    @property
    def iv(self, window: int = 20) -> float:
        product=self.product
        prices = self._cls_fair(product)
        if len(prices) >= window + 1:
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(-window - 1, -1)]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            return std_dev * (len(returns) ** 0.5)  # approximate scaling
        return 0.16  # fallback
    

    @property
    def hv(self, window: int = 20) -> float:
        product=self.product
        prices = self._cls_fair(product)
        if len(prices) >= window + 1:
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(-window - 1, -1)]
            squared_returns = [r ** 2 for r in returns]
            hv = sum(squared_returns) / len(squared_returns)
            return hv ** 0.5
        return 0.16  # fallback

class Strategy:
    @staticmethod
    def improve3(state: Status,
                 window: int = 20,
                 long_window_mult: int = 3,
                 iv_threshold: float = 0.2,
                 z_threshold: float = 1.5,
                 spread_threshold: int = 1,
                 trade_size_fraction: float = 0.3,
                 slope_window: int = 5,      # **ADDED: Window for MA slope calc**
                 max_ma_slope: float = 0.1): # **ADDED: Threshold for MA slope**
        """
        Further Corrected: Uses MA slope as a stronger trend filter.
        Disables trading if the long-term MA slope is too steep.

        Args:
            # ... (previous args) ...
            slope_window (int): How many steps back to compare long MA for slope calc.
            max_ma_slope (float): Max absolute slope allowed for long MA to permit trading.

        Returns:
            List[Order]: A list of orders to be placed.
        """
        orders: List[Order] = []
        long_window = window * long_window_mult

        # --- Data Availability Checks ---
        historical_fair_prices = state.fair_price
        # Need enough data for long window + slope lookback
        if len(historical_fair_prices) < long_window + slope_window:
            return orders

        # --- Calculate Metrics ---
        # Get prices needed for current short/long MA and previous long MA
        all_prices = np.array(historical_fair_prices[-(long_window + slope_window):], dtype=float)
        all_prices = all_prices[~np.isnan(all_prices)] # Clean NaNs once

        # Ensure enough valid data points remain after cleaning
        if len(all_prices) < max(long_window // 2, window // 2) + slope_window:
             return orders

        # Calculate current short-term mean/std/price
        prices_short = all_prices[-window:]
        if len(prices_short) < max(2, window // 2): return orders # Check again for short window slice
        mean_price_short = np.mean(prices_short)
        std_price_short = np.std(prices_short)
        current_price = prices_short[-1]

        if std_price_short == 0: return orders

        # Calculate current long-term mean
        prices_long_current = all_prices[-long_window:]
        if len(prices_long_current) < max(2, long_window // 2): return orders # Check long slice
        mean_price_long_current = np.mean(prices_long_current)

        # Calculate previous long-term mean for slope
        prices_long_prev = all_prices[-(long_window + slope_window):-slope_window]
        if len(prices_long_prev) < max(2, long_window // 2): return orders # Check prev long slice
        mean_price_long_prev = np.mean(prices_long_prev)

        # --- Trend Filter (MA Slope) ---
        long_ma_slope = (mean_price_long_current - mean_price_long_prev) / slope_window
        if abs(long_ma_slope) > max_ma_slope:
             # Slope too steep (strong trend), disable mean-reversion trades
             return orders

        # --- Get Current VWAP and Spread ---
        current_vwap = state.vwap
        current_spread = state.bid_ask_spread
        if current_vwap is None or current_spread is None: return orders

        # --- Volatility and Spread Filters ---
        try:
            current_iv = state.iv
            if current_iv > iv_threshold: return orders
        except Exception: return orders

        if current_spread > spread_threshold: return orders

        # --- Z-Score Signal ---
        zscore = (current_price - mean_price_short) / std_price_short
        if abs(zscore) < z_threshold: return orders

        # --- Trade Execution Logic (using previous comparison filter as well) ---

        # Buy Logic: z < threshold, price < vwap, AND price < long MA (current one)
        if zscore < -z_threshold and current_price < current_vwap and current_price < mean_price_long_current:
            available_buy_volume = state.possible_buy_amt
            if available_buy_volume > 0:
                best_ask_price = state.best_ask
                if best_ask_price is not None:
                    size = max(1, min(int(available_buy_volume * trade_size_fraction), available_buy_volume))
                    if size > 0:
                         orders.append(Order(state.product, int(best_ask_price), size))
                         state.rt_position_update(state.rt_position + size)

        # Sell Logic: z > threshold, price > vwap, AND price > long MA (current one)
        elif zscore > z_threshold and current_price > current_vwap and current_price > mean_price_long_current:
             available_sell_volume = state.possible_sell_amt
             if available_sell_volume > 0:
                 best_bid_price = state.best_bid
                 if best_bid_price is not None:
                      size = max(1, min(int(available_sell_volume * trade_size_fraction), available_sell_volume))
                      if size > 0:
                          orders.append(Order(state.product, int(best_bid_price), -size))
                          state.rt_position_update(state.rt_position - size)

        return orders



    @staticmethod
    def improve2(state: Status, window: int = 20, num_std: float = 2.0, iv_threshold: float = 0.3):
        orders = []

        # Require enough data
        if len(state.fair_price) < window:
            return orders

        # Use Kalman filtered mean if available
        mean_price = state.kalman_filtered_mean if hasattr(state, "kalman_filtered_mean") else np.mean(state.fair_price_history[-window:])
        std_price = np.std(state.fair_price[-window:])
        current_price = state.fair_price[-1]

        upper_band = mean_price + num_std * std_price
        lower_band = mean_price - num_std * std_price

        # Avoid trading in high volatility
        if state.iv > iv_threshold:
            return orders

        # Filter out noisy range trading (i.e., too close to mean)
        if abs(current_price - mean_price) < 0.2 * std_price:
            return orders

        # Avoid if spread is too high
        if state.bid_ask_spread > 2:
            return orders

        # Buy logic
        if current_price < lower_band and state.rt_position < state.position_limit:
            strength = (mean_price - current_price) / std_price
            size = min(state.possible_buy_amt, int(strength * state.position_limit))
            if size > 0:
                orders.append(Order(state.product, state.best_ask, size))
                executed = min(size, state.total_askamt)
                state.rt_position_update(state.rt_position + executed)

        # Sell logic
        elif current_price > upper_band and state.rt_position > -state.position_limit:
            strength = (current_price - mean_price) / std_price
            size = min(state.possible_sell_amt, int(strength * state.position_limit))
            if size > 0:
                orders.append(Order(state.product, state.best_bid, -size))
                executed = min(size, state.total_bidamt)
                state.rt_position_update(state.rt_position - executed)

        return orders
    
    @staticmethod
    def improved(state: Status, window: int = 20, num_std: float = 2.0):
        orders = []

        # Use fair price history
        prices = state.fair_price
        if len(prices) < window:
            return orders  # not enough data

        recent = prices[-window:]
        mean_price = np.mean(recent)
        std_price = np.std(recent)

        upper_band = mean_price + num_std * std_price
        lower_band = mean_price - num_std * std_price
        current_price = prices[-1]

        # Ensure we don't get eaten by spread
        if state.bid_ask_spread > 2:
            return orders

        # Buy if price is significantly low
        if current_price < lower_band and state.rt_position < state.position_limit:
            buy_amount = state.possible_buy_amt
            orders.append(Order(state.product, state.best_ask, buy_amount))
            executed = min(buy_amount, state.total_askamt)
            state.rt_position_update(state.rt_position + executed)

        # Sell if price is significantly high
        elif current_price > upper_band and state.rt_position > -state.position_limit:
            sell_amount = state.possible_sell_amt
            orders.append(Order(state.product, state.best_bid, -sell_amount))
            executed = min(sell_amount, state.total_bidamt)
            state.rt_position_update(state.rt_position - executed)

        return orders
    
    @staticmethod
    def basic(state:Status):
        orders=[]
        bd1=state.hist_order_depth("bidprc",depth=1,size=100)
        bd2=state.hist_order_depth("bidprc",depth=2,size=100)
        as1=state.hist_order_depth("askprc",depth=1,size=100)
        as2=state.hist_order_depth("askprc",depth=2,size=100)
        mean1=np.mean(bd1[-30:])
        mean2=np.mean(bd1[-80:])
        mean3=np.mean(as1[-30:])
        mean4=np.mean(as1[-80:])
        # currect_price=state.maxamt_midprc
        if(mean1>mean2):
            buy_amount = state.possible_buy_amt
            orders.append(Order(state.product, state.best_ask, buy_amount))
            executed_amount = min(buy_amount, state.total_askamt)
            state.rt_position_update(state.rt_position + executed_amount)

        elif(mean3<mean4):
            sell_amount = state.possible_sell_amt
            orders.append(Order(state.product, state.best_bid, -sell_amount))
            executed_amount = min(sell_amount, state.total_bidamt)
            state.rt_position_update(state.rt_position - executed_amount)

        return orders
    
    @staticmethod
    def bollinger(state: Status, window: int = 20, num_std_dev: float = 2.0):
        orders = []
        num_std_dev=state.worst_ask-state.best_bid

        # Ensure enough price history
        if len(state.fair_price) < window:
            return orders

        prices = state.fair_price[-window:]
        mean = np.mean(prices)
        std = np.std(prices)

        upper_band = mean + num_std_dev * std
        lower_band = mean - num_std_dev * std
        current_price = state.maxamt_midprc

        # Buy when price is below lower band
        if current_price < lower_band:
            buy_amount = state.possible_buy_amt
            orders.append(Order(state.product, state.best_ask, buy_amount))
            executed_amount = min(buy_amount, state.total_askamt)
            state.rt_position_update(state.rt_position + executed_amount)

        # Sell when price is above upper band
        elif current_price > upper_band:
            sell_amount = state.possible_sell_amt
            orders.append(Order(state.product, state.best_bid, -sell_amount))
            executed_amount = min(sell_amount, state.total_bidamt)
            state.rt_position_update(state.rt_position - executed_amount)

        return orders


    @staticmethod
    def mm_glft(
        state: Status,
        fair_price,
        mu=0,
        sigma=0.45,
        gamma=1e-9,
        order_amount=1000,
    ):
        
        q = state.rt_position / order_amount
        #Q = state.position_limit / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)

        A_b = 0.25
        A_a = 0.25

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) + (-mu / (gamma * sigma**2) + (2 * q + 1) / 2) * math.sqrt((sigma**2 * gamma) / (2 * kappa_b * A_b) * (1 + gamma / kappa_b)**(1 + kappa_b / gamma))
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + (mu / (gamma * sigma**2) - (2 * q - 1) / 2) * math.sqrt((sigma**2 * gamma) / (2 * kappa_a * A_a) * (1 + gamma / kappa_a)**(1 + kappa_a / gamma))

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(p_b, fair_price) # Set the buy price to be no higher than the fair price to avoid losses
        p_b = min(p_b, state.best_bid + 1) # Place the buy order as close as possible to the best bid price
        p_b = max(p_b, state.maxamt_bidprc + 1) # No market order arrival beyond this price

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        orders = []
        if buy_amount > 0:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders
    
    @staticmethod
    def vol_arb(option: Status,threshold=0.0018):

        vol_spread = option.iv - option.hv

        orders = []

        if vol_spread > threshold:
            sell_amount = option.possible_sell_amt
            orders.append(Order(option.product, option.best_bid, -sell_amount))
            executed_amount = min(sell_amount, option.total_bidamt)
            option.rt_position_update(option.rt_position - executed_amount)

        elif vol_spread < -threshold:
            buy_amount = option.possible_buy_amt
            orders.append(Order(option.product, option.best_ask, buy_amount))
            executed_amount = min(buy_amount, option.total_askamt)
            option.rt_position_update(option.rt_position + executed_amount)

        return orders
    
    @staticmethod
    def mean_strat(state:Status,lookback:int):
        orders=[]
        mid_price=state.hist_order_depth("bidprc",depth=1,size=10)
        # mid_price=np.array(state.maxamt_midprc)
        if len(mid_price)<lookback:
            return orders
        last=mid_price[-lookback:]
        first_derivative=np.mean(np.gradient(last))
        second_derivative=np.mean(np.gradient(np.gradient(last)))
        diff=len(last)
        mean=last[0]+diff*first_derivative+diff*diff*second_derivative*0.5

        if mean>mid_price[0]:
            buy_amount = state.possible_buy_amt
            orders.append(Order(state.product, state.best_ask, buy_amount))
            executed_amount = min(buy_amount, state.total_askamt)
            state.rt_position_update(state.rt_position + executed_amount)

        if mean<mid_price[0]:
            sell_amount = state.possible_sell_amt
            orders.append(Order(state.product, state.best_bid, -sell_amount))
            executed_amount = min(sell_amount, state.total_bidamt)
            state.rt_position_update(state.rt_position - executed_amount)

        return orders
    

    @staticmethod
    def kalman(state:Status):
        orders = []
        
        # Update Kalman Filter with current price
        threshold=25
        state.update_kalman_mean(state.maxamt_midprc)
        current_price = state.maxamt_midprc
        mean = state.kalman_filtered_mean
        std = state.iv  # Use implied volatility as a proxy for price volatility
        
        # Calculate z-score
        if std > 0:
            z_score = (current_price - mean) / std
        else:
            z_score = 0

        # Buy if price is significantly below estimated mean
        if z_score < -threshold:
            buy_amount = state.possible_buy_amt
            orders.append(Order(state.product, state.best_ask, buy_amount))
            executed_amount = min(buy_amount, state.total_askamt)
            state.rt_position_update(state.rt_position + executed_amount)

        # Sell if price is significantly above estimated mean
        elif z_score > threshold:
            sell_amount = state.possible_sell_amt
            orders.append(Order(state.product, state.best_bid, -sell_amount))
            executed_amount = min(sell_amount, state.total_bidamt)
            state.rt_position_update(state.rt_position - executed_amount)

        return orders
    
    






class Trade:

    # @staticmethod   
    # def amethysts(state: Status) -> list[Order]:

    #     current_price = state.maxamt_midprc

    #     orders = []
        # orders.extend(Strategy.arb(state=state, fair_price=current_price))
    #     orders.extend(Strategy.mm_ou(state=state, fair_price=current_price, gamma=0.1, order_amount=20))

    #     return orders


    @staticmethod   
    def squid(state: Status) -> list[Order]:
        current_price = state.maxamt_midprc

        orders = []
        # best till now
        # orders.extend(Strategy.bollinger(state=state))
        # orders.extend(Strategy.improved(state=state))
        orders.extend(Strategy.improve3(state=state))
        # orders.extend(Strategy.vol_arb(option=state))  
        # orders.extend(Strategy.mean_strat(state=state,lookback=10))
        # orders.extend(Strategy.kalman(state=state))
        # orders.extend(Strategy.mm_glft(state=state,fair_price=current_price))

        return orders
    

class Trader:

    # state_amethysts = Status('AMETHYSTS')
    state_squid = Status('SQUID_INK')

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state)

        result = {}
        conversions = 0
        traderData = ""

        # round 1
        # result["AMETHYSTS"] = Trade.amethysts(self.state_amethysts)
        result["SQUID_INK"]= Trade.squid(self.state_squid)
        

        # traderData = "SAMPLE" 
        # logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    


    # testing script
    # prosperity3bt ou.py 1 --data data/ --no-out --print --merge-pnl