import pandas as pd
import numpy as np
import QuantLib as ql
from datetime import timedelta, date, datetime
import uuid

params_irs = {
    'start_date': '02/01/2013',
    'n_trades_day': 100,            # expected average number of trades per day
    'n_trades_day_stddev': 20,      # standard devation of daily number of trades to be used in randomization
    'counterparties': 1000,         # the number of counterparties
    'currencies': {                 # default start date adjustment T + x
        'EUR': 2,
        'GBP': 0,
        'USD': 2
    },
    'tenure': list(range(1, 10)),         # all possible tenures in years
    'leg': ['Fixed', 'Float'],      # different leg types
    'float_rate': ['3M', '6M', '9M', '12M'],
    'daycount': ['act/360'],        # different day counts that can be applied
    'notional_min': 1000000,        # the lowest notional
    'notional_max': 150000000,      # the highest notional
    'notional_avg': 10000000,       # the average notional
    'notional_stddev': 3000000,     # standard deviation of notional to be used in randomization
}

columns = [
    'TradeId',
    'Trade Date',
    'Counterparty A',
    'Counterparty B',
    'Notional',
    'Currency',
    'Start Date',
    'Maturity Date',
    'Tenure',
    'Receive',
    'Pay',
    'Fixed Rate',
    'Float Rate',
    'Rate Conv Fixed',
    'Rate Conv Float',
]


def daterange(start_date, end_date, calendar):
    for n in range(int((end_date - start_date).days)):
        d = start_date + timedelta(n)
        if calendar.isBusinessDay(ql.Date(d.day, d.month, d.year)):
            yield start_date + timedelta(n)


def adjust_start_date(data):
    uk_calendar = ql.UnitedKingdom()
    adj_dates = [uk_calendar.advance(ql.Date.from_date(x[0]),
                                     ql.Period(params_irs['currencies'][x[1]], ql.Days),
                                     ql.Following).to_date()
                 for x in data[['Trade Date', 'Currency']].values]
    return adj_dates


def generate_maturity_dates(data):
    uk_calendar = ql.UnitedKingdom()

    ql_maturity_dates = [
        uk_calendar.advance(ql.Date.from_date(x[0]), ql.Period(int(x[1] * 12), ql.Months), ql.Following).to_date()
        for x in data[['Start Date', 'Tenure']].values
    ]

    return ql_maturity_dates


def generate_trades(data):
    n_rows = data.shape[0]

    print('Generating Counterparty A')
    data['Counterparty A'] = 'Counterparty 0'

    print('Generating Counterparty B')
    data['Counterparty B'] = ['Counterparty ' + str(x)
                              for x in np.random.randint(1, params_irs['counterparties'] + 1, n_rows)]

    print('Generating Notional')
    data['Notional'] = [int(x)
                        for x in np.random.normal(params_irs['notional_avg'],
                                                  params_irs['notional_stddev'],
                                                  n_rows)]
    data['Notional'] = data['Notional'] - data['Notional'] % 10000
    data['Notional'] = data['Notional'].astype('int64')

    print('Generating Currency')
    data['Currency'] = [list(params_irs['currencies'].keys())[x]
                        for x in np.random.randint(0, len(params_irs['currencies']), n_rows)]

    print('Generating Start Date')
    data['Start Date'] = adjust_start_date(data)
    data['Start Date'] = data['Start Date'].astype('datetime64')

    print('Generating Tenure')
    data['Tenure'] = data['Tenure'].astype('float64')
    data['Tenure'] = [x / 4 for x in np.random.randint(1, len(params_irs['tenure']) * 4, n_rows)]

    print('Generating Maturity Date')
    data['Maturity Date'] = generate_maturity_dates(data)
    data['Maturity Date'] = data['Maturity Date'].astype('datetime64')

    types = [x for x in np.random.randint(0, 2, n_rows)]
    print('Generating Receive type')
    data['Receive'] = [params_irs['leg'][x] for x in types]

    print('Generating Pay type')
    data['Pay'] = [params_irs['leg'][1 - x] for x in types]

    print('Generating Float Rates')
    data['Float Rate'] = [params_irs['float_rate'][x]
                          for x in np.random.randint(0, len(params_irs['float_rate']), n_rows)]

    print('Generating Rate Conventions')
    data['Rate Conv Fixed'] = [params_irs['daycount'][x]
                               for x in np.random.randint(0, len(params_irs['daycount']), n_rows)]
    data['Rate Conv Float'] = [params_irs['daycount'][x]
                               for x in np.random.randint(0, len(params_irs['daycount']), n_rows)]

    # remove all trades that already matured
    today = date.today()
    data = data[(data['Maturity Date'].dt.date - today).dt.days > 0]

    return data


def main():

    date_from = datetime.strptime(params_irs['start_date'], '%d/%m/%Y').date()
    date_to = (datetime.today() - timedelta(1)).date()

    # generate trade days
    trade_days = [x for x in daterange(date_from, date_to, ql.UnitedKingdom())]

    # generate number of trades each day
    num_trades_each_day = np.round(
        np.random.normal(params_irs['n_trades_day'], params_irs['n_trades_day_stddev'], len(trade_days)),
        0).astype(int)

    # generate unique trade ids
    trade_ids_dates = [(uuid.uuid1().hex, trade_days[i])
                       for i in range(len(num_trades_each_day))
                       for j in range(num_trades_each_day[i])]

    # create pandas DataFrame to store information
    data = pd.DataFrame(index=list(zip(*trade_ids_dates))[0], columns=columns).drop('TradeId', axis=1)
    data.index.rename('TradeId', inplace=True)

    # populate trade dates
    data['Trade Date'] = list(zip(*trade_ids_dates))[1]
    data['Trade Date'] = data['Trade Date'].astype('datetime64')

    print('{} trades will be generated'.format(np.sum(num_trades_each_day)))

    # populate data for every trade
    data = generate_trades(data)

    print(data.dtypes)

    # save data into CSV file
    data.to_csv('data.csv')


if __name__ == '__main__':
    main()
