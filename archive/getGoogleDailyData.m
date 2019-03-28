function [data, dividends] = getGoogleDailyData(tickers, startDate, endDate, dateFormat)
% GETGOOGLEDAILYDATA scrapes the Google Finance website for one or more
% ticker symbols and returns OHLC, volume, adjusted close, and dividend
% information for the date range specified.
%
% Inputs: 
% tickers: Either a character string or cell array of strings listing one
%   or more (Google-formatted) ticker symbols
% startDate, endDate: Either MATLAB datenums or character strings listing
%   the date range desired (inclusive)
% dateFormat (optional): If startDate and endDate are strings, then this
%   indicates their format.  (See the 'formatIn' argument of DAMENUM.)
%
% Outputs:
% data: A structure containing fields for each ticker requested.  The
%   fields are named by genvarname(ticker).  Each field is an N-by-7 table
%   listing the dates, open, high, low, close, volume, and adjusted close
%   for all N of the trading days between startDate and endDate and in
%   increasing order of date.
% dividends: A structure containing fields for each ticker requested.  The
%   fields are named by genvarname(ticker).  Each field is an M-by-2 table
%   listing the ex-dividend dates and the dividend amounts for each ticker
%   over the date range from startDate to endDate.
%
% NOTE: If the version of MATLAB is less than 8.2 (R2013b), then data and
%   dividends returns structures of dataset arrays (Statistics Toolbox
%   required).
%
% EXAMPLE:
% [data, dividends] = getGoogleDailyData({'NASDAQ:MSFT', 'EPA:ML'},'01/01/2010', '01/01/2013', 'dd/mm/yyyy');

%% 1. Input parsing
% Check to see if a single ticker was provided as a string; if so, make it
% a cell array to better fit the later logic.
if ischar(tickers)
    tickers = {tickers};
end

% If dateFormat is provided, use it.  Otherwise, assume datenums were
% given.
if nargin == 4
    startDate = datenum(startDate, dateFormat);
    endDate   = datenum(  endDate, dateFormat);
end

% Prepare the URL for the historical scrapes.  Note that we have to load
% all data up to today in order to properly calculate adjusted closing
% prices, even if the endDate isn't today.  We'll trim off any extras
% later.
url1 = ['http://www.google.com/finance/historical?startdate=' ...
    datestr(startDate, 'mmm+dd,+yyyy') '&enddate=' ...
    datestr(now, 'mmm+dd,+yyyy') '&num=200&q='];

% Determine if we're using tables or datasets:
isBeforeR2013b = verLessThan('matlab', '8.2');

%% 2. Load Data in a loop
h = waitbar(0, 'Getting daily data from Google');
for iTicker = 1:length(tickers)
    
    %% 2a. Go to ticker's main page to read dividend information
    % Google does not adjust for dividends, although it does adjust for stock
    % splits.  We need to load and scrape these from the main page.
    www1 = urlread(['http://www.google.com/finance?q=' tickers{iTicker}]);
    rawDividends = regexp(www1, '(?<=\\"dividend\\",).*?(?=},)', 'match');
    dates = regexp(rawDividends, '(?<="_date\\":\\")[0-9.]*?(?=\\)', 'match', 'once');
    amounts = regexp(rawDividends, '(?<=amount\\":\\")[0-9.]*?(?=\\)', 'match', 'once');
    
    % Google tends to set the dividend times as some point in the afternoon
    % or morning of the ex-dividend date.  To better line up with
    % convention, we simply FLOOR it.
    dates = floor(datenum('01-Jan-1970') + str2double(dates')/(24*60*60));
    amounts = str2double(amounts');
    
    if isBeforeR2013b
        div = dataset(dates, amounts, 'VarNames', {'Date', 'Amount'});
    else
        div = table(dates, amounts, 'VariableNames', {'Date', 'Amount'});
    end
    
    %% 2b. Scrape Date, OHLC, and Volume from Google history pages
    % These are only presented in HTML tables shown 200 business days at a
    % time at most. 
    iBlock = 0;
    isFinished = false;
    history = [];
    
    while ~isFinished
        www2 = urlread([url1 tickers{iTicker} '&start=' num2str(iBlock)]);
        www2 = regexp(www2, '(?<=>Volume\n<tr>\n).*(?=</table>)', 'match', 'once');
        block = reshape(regexp(www2, '(?<=>)[-a-zA-Z0-9,. ]+(?=\n)', 'match'), 6, [])';
        if ~isempty(block)
            history = [history; block]; %#ok<AGROW>
            iBlock = iBlock + 200;
        else
            isFinished = true;
        end
    end
    
    % Special behaviour if history comes back empty: this means that no
    % price info was returned.  Error and say which asset is invalid:
    if isempty(history)
        close(h)
        error('getGoogleDailyData:invalidTicker', ...
            ['No data returned for ticker ''' tickers{iTicker} '''. Is this a valid symbol?'])
    end
    
    if isBeforeR2013b
        ds = dataset(datenum(history(:,1), 'mmm dd, yyyy'), ...
            str2double(history(:,2)), str2double(history(:,3)), ...
            str2double(history(:,4)), str2double(history(:,5)), ...
            str2double(history(:,6)), 'VarNames', ...
            {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'});
    else
        ds = table(datenum(history(:,1), 'mmm dd, yyyy'), ...
            str2double(history(:,2)), str2double(history(:,3)), ...
            str2double(history(:,4)), str2double(history(:,5)), ...
            str2double(history(:,6)), 'VariableNames', ...
            {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'});
    end
    
    %% 2c. Calculate adjusted closing prices
    % Following the Yahoo! datafeed and CRSP documentation, we calculate
    % the adjustment factors for dividends ONLY (since Google already
    % adjusts for splits), multiply them by the unadjusted closing prices,
    % and then round to the nearest cent.
    ds.AdjClose = ...
        calculateAdjustedClose(ds.Date, ds.Close, div.Date, div.Amount);
            
    ds = flipud(ds);
    
    % Trim off any extra dates if endDate isn't today:
    ds(  ds.Date > endDate, :) = [];
    div(div.Date > endDate, :) = [];
    
    data.(     genvarname(tickers{iTicker})) = ds;
    dividends.(genvarname(tickers{iTicker})) = div;
    
    waitbar(iTicker/length(tickers), h);
end

close(h)