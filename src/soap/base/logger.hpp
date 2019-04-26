#ifndef _SOAP_LOGGER_H
#define	_SOAP_LOGGER_H

#include <sstream>
#include <iostream>

namespace soap {



enum TLogLevel {logERROR, logWARNING, logINFO, logDEBUG};

#define LOGIF(level, log) \
if ( &log != NULL && level > (log).getReportLevel() ) ; \
else (log)(level)

#define LOG(plog) \
if (plog == NULL) ; \
else (*plog)()

class LogBuffer : public std::stringbuf {

public:
	LogBuffer() : std::stringbuf(), _silent(false),
                _errorPreface("ERROR "), _warnPreface("WARNING "),
                _infoPreface(""), _dbgPreface("DEBUG "),
                _writePreface(true) {}

    void silence() { _silent = true; }
    void setSilent(bool set_silent) { _silent = set_silent; }
    void toggleSilence() { _silent = !_silent; }

        // sets the log level (needed for output)
	void setLogLevel(TLogLevel LogLevel) { _LogLevel = LogLevel; }

        // sets Multithreading (buffering required)
        void setMultithreading( bool maverick ) { _maverick = maverick; }

        // sets preface strings for logERROR, logWARNING, ...
        void setPreface(TLogLevel level, std::string preface) {
            switch ( level )
            {

                case logERROR:
                    _errorPreface = preface;
                    break;
                case logWARNING:
                    _warnPreface = preface;
                    break;
                case logINFO:
                    _infoPreface = preface;
                    break;
                case logDEBUG:
                    _dbgPreface = preface;
                    break;
            }
        }

        void EnablePreface() { _writePreface = true; }
        void DisablePreface() { _writePreface = false; }

        // flushes all collected messages
        void FlushBuffer(){ std::cout << _stringStream.str(); _stringStream.str(""); }

        // returns the pointer to the collected messages
        std::string Messages() {
            std::string _messages = _stringStream.str();
            _stringStream.str("");
            return _messages;
        }

private:

  // Log Level (WARNING, INFO, etc)
  TLogLevel _LogLevel;

  // temporary buffer to store messages
  std::ostringstream _stringStream;

  // Multithreading
  bool _maverick;
  bool _writePreface;
  bool _silent;

  std::string _timePreface;
  std::string _errorPreface;
  std::string _warnPreface;
  std::string _infoPreface;
  std::string _dbgPreface;


protected:
	virtual int sync() {
            if (_silent) {
                str("");
                return 0;
            }

            std::ostringstream _message;

            if (_writePreface) {
                switch ( _LogLevel )
                {
                    case logERROR:
                        _message << _errorPreface;
                        break;
                    case logWARNING:
                        _message << _warnPreface;
                        break;
                    case logINFO:
                        _message << _infoPreface;
                        break;
                    case logDEBUG:
                        _message << _dbgPreface;
                        break;
                }
            }

            if ( !_maverick ) {
                // collect all messages of one thread
                _stringStream << _message.str()  << "" << str();
            } else {
                // if only one thread outputs, flush immediately
                std::cout << _message.str() << "" << str() << std::flush;
            }
            _message.str("");
	    str("");
	    return 0;
	}

};

class Logger : public std::ostream {

       friend std::ostream& operator<<( std::ostream& out, Logger&  logger ) {
           out << logger.Messages();
           return out;
       }

public:
	Logger( TLogLevel ReportLevel) : std::ostream(new LogBuffer()) {
            _ReportLevel = ReportLevel;
            _maverick = true;
            _verbose = false;
     }

	 Logger() : std::ostream(new LogBuffer()) {
		 _ReportLevel = logDEBUG;
		 _maverick = true;
		 dynamic_cast<LogBuffer *>( rdbuf() )->setMultithreading(_maverick);
	 }

	~Logger() {
            //dynamic_cast<LogBuffer *>( rdbuf())->FlushBuffer();
            delete rdbuf();
            rdbuf(NULL);
	}

	Logger &operator()( TLogLevel LogLevel = logINFO) {
		//rdbuf()->pubsync();
		dynamic_cast<LogBuffer *>( rdbuf() )->setLogLevel(LogLevel);
		return *this;
	}

        void setReportLevel( TLogLevel ReportLevel ) { _ReportLevel = ReportLevel; }
        void silence() {
            _silent = true;
            dynamic_cast<LogBuffer*>(rdbuf())->silence();
        }
        void setSilent(bool set_silent) {
            _silent = set_silent;
            dynamic_cast<LogBuffer*>(rdbuf())->setSilent(set_silent);
        }
        bool isSilent() { return _silent; }
        void toggleSilence() {
            _silent = !_silent;
            dynamic_cast<LogBuffer*>(rdbuf())->toggleSilence();
        }
        void setVerbose(bool verbose) {
            _verbose = verbose;
        }
        bool verbose() { return _verbose; }
        void setMultithreading( bool maverick ) {
            _maverick = maverick;
            dynamic_cast<LogBuffer *>( rdbuf() )->setMultithreading( _maverick );
        }
        bool isMaverick() { return _maverick; }

        TLogLevel getReportLevel( ) { return _ReportLevel; }

        void setPreface(TLogLevel level, std::string preface) {
            dynamic_cast<LogBuffer *>( rdbuf() )->setPreface(level, preface);
        }

        void EnablePreface() {
            dynamic_cast<LogBuffer *>( rdbuf() )->EnablePreface();
        }

        void DisablePreface() {
            dynamic_cast<LogBuffer *>( rdbuf() )->DisablePreface();
        }

private:
    // at what level of detail output messages
    TLogLevel _ReportLevel;

    // if true, only a single processor job is executed
    bool      _maverick;
    bool      _silent;
    bool      _verbose;

    std::string Messages() {
        return dynamic_cast<LogBuffer *>( rdbuf() )->Messages();
    }

};

/**
*   \brief Timestamp returns the current time as a string
*  Example: cout << TimeStamp()
*/
class TimeStamp
{
  public:
    friend std::ostream & operator<<(std::ostream &os, const TimeStamp& ts)
    {
        time_t rawtime;
        tm * timeinfo;
        time(&rawtime);
        timeinfo = localtime( &rawtime );
        os  << (timeinfo->tm_year)+1900
            << "-" << timeinfo->tm_mon + 1
            << "-" << timeinfo->tm_mday
            << " " << timeinfo->tm_hour
            << ":" << timeinfo->tm_min
            << ":"  << timeinfo->tm_sec;
         return os;
    }

    explicit TimeStamp() {};

};

}

#endif
