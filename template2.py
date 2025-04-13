import numpy as np
import math
from statistics import NormalDist
from datamodel import *
from typing import List, Dict, Tuple, Any
import json

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


def reg_slope(x:List,Y:List)->float:
    mn=min(50,len(Y))

    X=np.array(x[-mn:])
    y=np.array(Y[-mn:])

    x_mean = np.mean(X)
    y_mean = np.mean(y)

    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    if(denominator==0):
        return 0
    m = numerator / denominator
    return m

def rsi(prices: list) -> float:
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100  # Avoid division by zero

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# No need to see please do not change
class Status:

    _position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
    }

    _state = None
    _realtime_position = {key:0 for key in _position_limit.keys()}
    # position of asset in real time

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
            'askamt3': [],
        } for product in _position_limit.keys()
    }

    _num_data = 0

    def __init__(self, product: str) -> None:
        """Initialize status object.

        Args:
            product (str): product

        """
        self.product = product
        self.ema_mid_35 = []
        self.ema_mid_100 = []
        self.reg_slope = []
        self.time = []
        self.rsi=[]
        self.rsi_ema=[]
        self.rain=0

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
            for prc, amt in sorted(orderdepth.sell_orders.items(), reverse=False):
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
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(prc)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(amt)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(np.nan)
                cnt += 1

        cls._num_data += 1
    
    def updates(self):
        # time update
        self.time.append(self.timestep*100)

        # reg
        self.reg_slope.append(reg_slope(self.hist_mid_prc(50),self.time))

        # RSI
        if(self.timestep>=50):
            self.rsi.append(rsi(self.hist_mid_prc(50)))
        else:
            self.rsi.append(0)

        # ema calc and update
        n35=2/51
        n100=2/2001
        if(len(self.ema_mid_35)==0):
            self.ema_mid_35.append(self.mid)
            self.rsi_ema.append(self.rsi[-1])
            self.ema_mid_100.append(self.mid)

        else:
            num=self.ema_mid_35[-1]*(1-n35)+n35*self.mid
            num2=self.ema_mid_100[-1]*(1-n100)+n100*self.mid
            num3=self.rsi_ema[-1]*(1-n35)+n35*self.rsi[-1]
            self.ema_mid_35.append(num)
            self.ema_mid_100.append(num2)
            self.rsi_ema.append(num3)

        
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
    def all_bids(self) -> list[tuple[int, int]]:
        """Return bid orders.

        Returns:
            dict[int, int].items(): bid orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].buy_orders.items())
    
    @property
    def all_asks(self) -> list[tuple[int, int]]:
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

    
    def hist_vwap_all(self, size:int) -> np.ndarray:
        res_array = np.zeros(size)
        volsum_array = np.zeros(size)
        for i in range(1,4):
            tmp_bid_vol = self.hist_order_depth(f'bidamt', i, size)
            tmp_ask_vol = self.hist_order_depth(f'askamt', i, size)
            tmp_bid_prc = self.hist_order_depth(f'bidprc', i, size)
            tmp_ask_prc = self.hist_order_depth(f'askprc', i, size)
            bid_nan_idx = np.isnan(tmp_bid_prc)
            ask_nan_idx = np.isnan(tmp_ask_prc)
            res_array = res_array + np.where(bid_nan_idx, 0, tmp_bid_prc*tmp_bid_vol) + np.where(ask_nan_idx, 0, -tmp_ask_prc*tmp_ask_vol)
            volsum_array = volsum_array + np.where(bid_nan_idx, 0, tmp_bid_vol) - np.where(ask_nan_idx, 0, tmp_ask_vol)
                
        return res_array / volsum_array

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
    def spread(self) -> int:
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
    def avg_bid(self)->int:
        sum=0
        vol=0
        for price,volume in self.all_bids:
            sum+=price*volume
            vol+=volume
        return sum/volume
    
    @property
    def avg_ask(self)->int:
        sum=0
        vol=0
        for price,volume in self.all_asks:
            sum+=price*volume
            vol+=volume
        return sum/volume

    @property
    def vwap_fair_price(self) -> float:
        """Calculate fair price based on volume-weighted average price of bids and asks."""
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders

        if not buy_orders or not sell_orders:
            return self.mid # Fallback to simple mid if book is empty/one-sided

        total_bid_value = sum(p * v for p, v in buy_orders.items())
        total_bid_volume = sum(buy_orders.values())

        total_ask_value = sum(p * abs(v) for p, v in sell_orders.items())
        total_ask_volume = sum(abs(v) for v in sell_orders.values())

        if total_bid_volume == 0 or total_ask_volume == 0:
            return self.mid # Fallback if one side has zero volume

        avg_bid = total_bid_value / total_bid_volume
        avg_ask = total_ask_value / total_ask_volume

        return (avg_bid + avg_ask) / 2.0


# All strategies here
class Strategy:
  
# example strategy
    @staticmethod
    def bollinger_band(state: Status, window: int = 20, num_std_dev: float = 2.0):
        orders = []
        num_std_dev=state.worst_ask-state.best_bid

        # Ensure enough price history
        mid_price=state.hist_mid_prc(window)
        if len(mid_price) < window:
            return orders

        prices = mid_price[-window:]
        mean = np.mean(prices)
        std = np.std(prices)

        upper_band = mean + num_std_dev * std
        lower_band = mean - num_std_dev * std
        current_price = state.maxamt_midprc

        # Buy when price is below lower band
        if current_price < lower_band:
            buy_amount = state.possible_buy_amt
            executed_amount = min(buy_amount, state.best_ask_amount)
            orders.append(Order(state.product, state.best_ask, executed_amount))
            state.rt_position_update(state.rt_position + executed_amount)

        # Sell when price is above upper band
        elif current_price > upper_band:
            sell_amount = state.possible_sell_amt
            executed_amount = min(sell_amount, state.best_bid_amount)
            orders.append(Order(state.product, state.best_bid, -executed_amount))
            state.rt_position_update(state.rt_position - executed_amount)

        return orders
    
    @staticmethod
    def fair_price_arbitrage(state:Status):
        fair=(state.avg_ask+state.avg_bid)/2
        orders=[]
        for price,amount in sorted(state.all_asks,reverse=True):
            if(price<=fair-1):
                buy_amount = state.possible_buy_amt
                executed_amount = min(buy_amount, abs(amount)) 
                state.last_price=price
                if(executed_amount>0):
                    orders.append(Order(state.product, price, executed_amount))
                    # print(orders[-1],state.rsi[-1],ex_low,ex_high,ad_low,ad_high)
                    state.rt_position_update(state.rt_position + executed_amount)

        for price,amount in sorted(state.all_bids,reverse=False):
                if(price>=fair+1):    
                    sell_amount = state.possible_sell_amt
                    executed_amount = min(sell_amount, amount)
                    state.last_price=price
                    if(executed_amount>0):
                        orders.append(Order(state.product, price, -executed_amount))
                        # print(orders[-1],state.rsi[-1],ex_low,ex_high,ad_low,ad_high)
                        state.rt_position_update(state.rt_position - executed_amount)
        return orders

    @staticmethod
    def hardcode(state:Status):
        orders=[]      
        for price,amount in sorted(state.all_asks,reverse=True):
            if(price<=10000):
                buy_amount = state.possible_buy_amt
                executed_amount = min(buy_amount, abs(amount)) 
                if(executed_amount>0):
                    if(price==9998):
                        state.rain+=executed_amount
                    orders.append(Order(state.product, price, executed_amount))
                    print(orders[-1],price,executed_amount)
                    state.rt_position_update(state.rt_position + executed_amount)

        for price,amount in sorted(state.all_bids,reverse=False):
            if(price>10000) or (state.rain>0 and price==10000):    
                sell_amount = state.possible_sell_amt
                executed_amount = min(sell_amount, amount)
                if(executed_amount>0):
                    if(price==10000):
                        state.rain-=executed_amount
                    orders.append(Order(state.product, price, -executed_amount))
                    print(orders[-1],price,-executed_amount)
                    state.rt_position_update(state.rt_position - executed_amount) 

        return orders
      
    @staticmethod
    def basic_strat(state:Status):
    #  working extremly well for squid ink 
    # profit analysis done
    # graph analysis done
    # risk management left
        orders=[]
        if(state.timestep<1000):
            return orders
        
        ad_low=np.percentile(state.rsi[-1000:],20)
        ad_high=np.percentile(state.rsi[-1000:],80)
        ex_high=np.percentile(state.rsi[-1000:],98)
        ex_low=np.percentile(state.rsi[-1000:],2)
        # ad_low=np.percentile(state.rsi_ema[-1000:],20)
        # ad_high=np.percentile(state.rsi_ema[-1000:],80)
        # ex_high=np.percentile(state.rsi_ema[-1000:],95)
        # ex_low=np.percentile(state.rsi_ema[-1000:],5)
        # buy
        if((len(state.ema_mid_35)>=2 and 
            state.ema_mid_35[-1]>=state.ema_mid_100[-1]+1 and state.ema_mid_35[-2]<state.ema_mid_100[-2] and 
            state.rsi[-1]>=ad_low) or (state.rsi[-1]<=ex_low)):
            for price,amount in sorted(state.all_asks,reverse=False):
                buy_amount = state.possible_buy_amt
                executed_amount = min(buy_amount, abs(amount)) 
                if(executed_amount>0):
                    orders.append(Order(state.product, price, executed_amount))
                    print(orders[-1])
                    state.rt_position_update(state.rt_position + executed_amount)

        # sell
        if  ((len(state.ema_mid_35)>=2 and 
            state.ema_mid_100[-1]>=state.ema_mid_35[-1]+1 and state.ema_mid_35[-2]>state.ema_mid_100[-2] and 
            state.rsi[-1]<=ad_high) or (state.rsi[-1]>=ex_high)):
            for price,amount in sorted(state.all_bids,reverse=True):
                sell_amount = state.possible_sell_amt
                executed_amount = min(sell_amount, amount)
                if(executed_amount>0):
                    orders.append(Order(state.product, price, -executed_amount))
                    print(orders[-1])
                    state.rt_position_update(state.rt_position - executed_amount)          

        return orders
      
    @staticmethod
    def kelp_adaptive_mm(state: Status) -> list[Order]:
        """Adaptive market making for KELP using EMA, inventory skew, and adaptive sizing."""
        orders = []
        product = state.product # Should be "KELP"
        limit = state.position_limit
        position = state.rt_position # Use real-time position for decisions

        # --- Parameters ---
        MIN_SPREAD = 2
        BASE_ORDER_SIZE = 15        # Max order size when near neutral inventory
        MIN_ORDER_SIZE = 3          # Minimum order size to place
        SPREAD_CAPTURE_RATIO = 0.4  # Target % of bid-ask spread to capture on each side
        INVENTORY_SKEW_FACTOR = 1.5 # How aggressively prices are skewed based on inventory (higher = more aggressive)
        SIZE_REDUCTION_FACTOR = 0.8 # How much max size reduces based on inventory (0 to 1)

        # --- Pre-computation & Checks ---
        if not state.all_bids or not state.all_asks: return []

        spread = state.best_ask - state.best_bid
        if spread < MIN_SPREAD: return []

        # Use the smoothed EMA fair price for KELP
        smoothed_fair_price = state.fair_price_ema_kelp
        if smoothed_fair_price is None: return [] # Wait for EMA warmup

        # --- Inventory Skew Calculation ---
        inventory_ratio = position / limit # Ranges from -1 to 1
        # Skew prices: If long (ratio > 0), lower both buy and sell prices. If short (ratio < 0), raise both.
        price_skew = inventory_ratio * (spread / 2.0) * INVENTORY_SKEW_FACTOR

        # --- Adaptive Size Calculation ---
        # Reduce max size as inventory moves away from zero
        size_penalty = abs(inventory_ratio) * BASE_ORDER_SIZE * SIZE_REDUCTION_FACTOR
        current_max_size = max(MIN_ORDER_SIZE, int(BASE_ORDER_SIZE - size_penalty))

        # --- Price Calculation ---
        # Calculate base offset from smoothed fair price
        price_offset = max(1, int(spread * SPREAD_CAPTURE_RATIO)) # Ensure at least 1 point offset

        # Calculate skewed buy/sell prices
        target_buy_price = int(smoothed_fair_price - price_offset - price_skew)
        target_sell_price = int(smoothed_fair_price + price_offset - price_skew) # Still subtract skew!

        # Ensure sell price is always >= buy price + 1 (basic sanity)
        target_sell_price = max(target_sell_price, target_buy_price + 1)

        # --- Volume Calculation ---
        buy_volume = min(current_max_size, state.possible_buy_amt)
        sell_volume = min(current_max_size, state.possible_sell_amt)

        # --- Order Placement ---
        if buy_volume >= MIN_ORDER_SIZE: # Only place if size meets minimum
            orders.append(Order(product, target_buy_price, buy_volume))
            state.rt_position_update(state.rt_position + buy_volume)
            # Optional Log: print(f"KELP: BUY {buy_volume}@{target_buy_price} (Skew:{price_skew:.2f}|Pos:{position})")


        # Re-check possible sell amount in case buy order changed rt_position significantly, although min() should handle it
        # sell_volume = min(current_max_size, state.possible_sell_amt) # Can recalculate if paranoid

        if sell_volume >= MIN_ORDER_SIZE: # Only place if size meets minimum
            orders.append(Order(product, target_sell_price, -sell_volume))
            state.rt_position_update(state.rt_position - sell_volume)
            # Optional Log: print(f"KELP: SELL {sell_volume}@{target_sell_price} (Skew:{price_skew:.2f}|Pos:{position})")


        return orders

        @staticmethod
        def resin_ema_mm(state: Status) -> list[Order]:
            """Market making based on EMA of VWAP fair value for RAINFOREST_RESIN."""
            orders = []
            product = state.product # Should be "RAINFOREST_RESIN"
            MIN_SPREAD = 2
            BASE_VOLUME_FACTOR = 15 # Base component of order size
            LIQUIDITY_SENSITIVITY = 20 # How much liquidity affects order size

            # Check if order book has both sides
            if not state.all_bids or not state.all_asks:
                # print(f"{product}: Missing bids or asks.") # Optional log
                return []

            spread = state.best_ask - state.best_bid
            if spread < MIN_SPREAD:
                # print(f"{product}: Spread too tight ({spread}), skipping.") # Optional log
                return []

            # EMA is updated in Trader.run *before* calling this strategy
            smoothed_price = state.fair_price_ema
            if smoothed_price is None:
                # print(f"{product}: EMA not yet calculated.") # Optional log
                return [] # Need EMA warmup

            # Determine Buy/Sell Prices based on EMA and spread
            # Using 0.48 factor from original code
            buy_price = int(smoothed_price - spread * 0.48)
            sell_price = int(smoothed_price + spread * 0.48)

            # Determine Volumes
            available_buy_liquidity = state.get_ask_volume_le(buy_price)
            available_sell_liquidity = state.get_buy_volume_ge(sell_price)
            limit_factor = min(state.total_bid_vol, state.total_ask_vol)
            # Avoid division by zero or extreme values if limit_factor is small
            liquidity_component = (limit_factor / 100.0) * LIQUIDITY_SENSITIVITY if limit_factor > 0 else 0
            base_volume = int(BASE_VOLUME_FACTOR + liquidity_component)

            # Calculate desired volume, considering base size and available liquidity at target price
            desired_buy_vol = max(base_volume, available_buy_liquidity)
            desired_sell_vol = max(base_volume, available_sell_liquidity)

            # Final volume capped by position limits
            buy_volume = max(0, min(desired_buy_vol, state.possible_buy_amt))
            sell_volume = max(0, min(desired_sell_vol, state.possible_sell_amt))

            # Place Orders (redundant limit checks removed, handled by possible_buy/sell_amt)
            if buy_volume > 0:
                orders.append(Order(product, buy_price, buy_volume))
                state.rt_position_update(state.rt_position + buy_volume)
                # print(f"{product}: BUY {buy_volume} @ {buy_price} | EMA: {smoothed_price:.2f}") # Optional log

            if sell_volume > 0:
                orders.append(Order(product, sell_price, -sell_volume))
                state.rt_position_update(state.rt_position - sell_volume)
                # print(f"{product}: SELL {sell_volume} @ {sell_price} | EMA: {smoothed_price:.2f}") # Optional log

            return orders


    @staticmethod
    def basic_strat2(state:Status):
    #  working extremly well for squid ink 
    # profit analysis done
    # graph analysis done
    # risk management left
        orders=[]
        if(state.timestep<1000):
            return orders
        
        ad_low=np.percentile(state.rsi[-1000:],30)
        ad_high=np.percentile(state.rsi[-1000:],80)
        ex_high=np.percentile(state.rsi[-1000:],98)
        ex_low=np.percentile(state.rsi[-1000:],2)
        # buy
        if((len(state.ema_mid_35)>=2 and 
            state.ema_mid_35[-1]>=state.ema_mid_100[-1]+1 and state.ema_mid_35[-2]<state.ema_mid_100[-2] and 
            state.rsi[-1]>state.rsi[-2] and state.rsi[-1]>=ad_low) or state.rsi[-1]<=ex_low):
            for price,amount in sorted(state.all_asks,reverse=False):
                buy_amount = state.possible_buy_amt
                executed_amount = min(buy_amount, abs(amount)) 
                if(executed_amount>0):
                    orders.append(Order(state.product, price, executed_amount))
                    # print(orders[-1],state.rsi[-1],ex_low,ex_high,ad_low,ad_high)
                    state.rt_position_update(state.rt_position + executed_amount)

        # sell
        if  ((len(state.ema_mid_35)>=2 and 
            state.ema_mid_100[-1]>=state.ema_mid_35[-1]+1 and state.ema_mid_35[-2]>state.ema_mid_100[-2] and 
            state.rsi[-1]<state.rsi[-2] and state.rsi[-1]<=ad_high) or state.rsi[-1]>=ex_high):
            for price,amount in sorted(state.all_bids,reverse=True):
                sell_amount = state.possible_sell_amt
                executed_amount = min(sell_amount, amount)
                if(executed_amount>0):
                    orders.append(Order(state.product, price, -executed_amount))
                    # print(orders[-1],state.rsi[-1],ex_low,ex_high,ad_low,ad_high)
                    state.rt_position_update(state.rt_position - executed_amount)          

        return orders
      
class Trade:

    @staticmethod   
    def SQUID_INK(state: Status) -> list[Order]:
        orders = []
        # all strategies here....
        orders.extend(Strategy.basic_strat(state=state))

        return orders
    
    @staticmethod   
    def KELP(state: Status) -> list[Order]:
        orders = []
        # all strategies here....
        orders.extend(Strategy.kelp_adaptive_mm(state=state))

        return orders
    
    @staticmethod   
    def RAINFOREST_RESIN(state: Status) -> list[Order]:
        orders = []
        # all strategies here....
        orders.extend(Strategy.hardcode(state=state))

        return orders


class Trader:
    # here initialise a status class for asset
    state_SQUID_INK = Status('SQUID_INK')
    state_RAINFOREST_RESIN = Status('RAINFOREST_RESIN')
    state_KELP = Status('KELP')
    state_CROISSANTS = Status('CROISSANTS')
    state_JAMS = Status('JAMS')
    state_DJEMBES = Status('DJEMBES')
    state_PICNIC_BASKET1 = Status('PICNIC_BASKET1')
    state_PICNIC_BASKET2 = Status('PICNIC_BASKET2')


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, Any]:
        Status.cls_update(state)
        
        result = {}
        conversions = 0
        traderData=''

        if "SQUID_INK" in state.order_depths.keys():
            self.state_SQUID_INK.updates()
            result["SQUID_INK"] = Trade.SQUID_INK(self.state_SQUID_INK)

        # if "KELP" in state.order_depths.keys():
        #     self.state_KELP.updates()
        #     result["KELP"] = Trade.KELP(self.state_KELP)

        # if "RAINFOREST_RESIN" in state.order_depths.keys():
        #     self.state_RAINFOREST_RESIN.updates()
        #     result["RAINFOREST_RESIN"] = Trade.RAINFOREST_RESIN(self.state_RAINFOREST_RESIN)


        # when testing in local backtest comment out all logger instances
        # logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
    


# testing script
# prosperity3bt template2.py 1-0 --data data/ --no-out --merge-pnl